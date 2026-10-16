"""Utility-aware spatial-memory scoring for the bundled SAM2/MedSAM2 attention.

The controller operates on the memory-unit spans emitted by ``SAM2Base``.  It
never changes memory contents or object-pointer tokens: it only returns an
additive bias for spatial-memory attention logits.  The same hook supports
leave-one-unit-out labels, no-op checks, attention analysis and the learned
utility head, keeping those comparisons on one implementation path.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


FEATURE_NAMES = (
    "current_global",
    "current_foreground",
    "memory_key",
    "temporal_distance",
    "conditioning",
    "predicted_iou",
    "object_score",
)


def _metadata_value(value, batch: int, device, dtype) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch, 1, device=device, dtype=dtype)
    if not torch.is_tensor(value):
        return torch.full((batch, 1), float(value), device=device, dtype=dtype)
    value = value.detach().to(device=device, dtype=dtype).reshape(value.shape[0], -1)
    if value.shape[0] == 1 and batch != 1:
        value = value.expand(batch, -1)
    return value.mean(dim=1, keepdim=True)


class UtilityHead(nn.Module):
    """Two-layer frame-level utility predictor from preregistered features."""

    def __init__(self, query_dim=256, memory_dim=64, hidden_dim=128):
        super().__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * query_dim + memory_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def input_dim(self):
        return self.mlp[0].in_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} utility features, got {features.shape[-1]}."
            )
        return self.mlp(features).squeeze(-1)


class UtilityController(nn.Module):
    """Build per-token attention biases while leaving object pointers untouched."""

    def __init__(self, head: UtilityHead, bias_scale: float = 1.0, learn_scale=True):
        super().__init__()
        self.head = head
        initial = torch.tensor(float(bias_scale)).log()
        if learn_scale:
            self.log_bias_scale = nn.Parameter(initial)
        else:
            self.register_buffer("log_bias_scale", initial)
        self.enabled = True
        self.excluded_frame_indices: set[int] = set()

    @property
    def bias_scale(self) -> torch.Tensor:
        return self.log_bias_scale.exp()

    def unit_features(self, curr, memory, units, query_context):
        batch = curr.shape[0]
        dtype, device = curr.dtype, curr.device
        current_global = query_context["global"].to(device=device, dtype=dtype)
        current_foreground = query_context["foreground"].to(device=device, dtype=dtype)
        rows = []
        for unit in units:
            memory_key = memory[:, unit["start"] : unit["end"]].mean(dim=1)
            metadata = torch.cat([
                _metadata_value(unit["temporal_distance"], batch, device, dtype),
                _metadata_value(unit["conditioning"], batch, device, dtype),
                _metadata_value(unit.get("predicted_iou"), batch, device, dtype),
                _metadata_value(unit.get("object_score"), batch, device, dtype),
            ], dim=-1)
            rows.append(torch.cat([
                current_global, current_foreground, memory_key, metadata
            ], dim=-1))
        return torch.stack(rows, dim=1)

    def forward(self, curr, memory, units, num_obj_ptr_tokens, query_context):
        del num_obj_ptr_tokens  # pointer tokens stay at zero bias by construction
        features = self.unit_features(curr, memory, units, query_context)
        utilities = self.head(features)
        token_bias = memory.new_zeros(memory.shape[0], memory.shape[1])
        if self.enabled:
            for index, unit in enumerate(units):
                token_bias[:, unit["start"] : unit["end"]] = (
                    self.bias_scale * utilities[:, index]
                ).unsqueeze(-1)
        for unit in units:
            if unit["frame_idx"] in self.excluded_frame_indices:
                token_bias[:, unit["start"] : unit["end"]] = -torch.inf
        # SDPA accepts [B, 1, 1, K], broadcast over heads and queries.
        attention_bias = token_bias[:, None, None, :]
        if not self.enabled and not self.excluded_frame_indices:
            attention_bias = None  # preserve the uninstrumented SDPA kernel path exactly
        return attention_bias, utilities, features.detach()


def attach_utility_controller(model, controller: UtilityController, capture_attention=False):
    """Attach to a built predictor and return the same model for fluent setup."""
    model.memory_attention.utility_controller = controller.to(model.device)
    model.memory_attention.capture_attention = capture_attention
    return model


def save_utility_head(controller: UtilityController, path: Path, metadata=None):
    payload = {
        "kind": "utility_memory_v1",
        "head": controller.head.state_dict(),
        "log_bias_scale": controller.log_bias_scale.detach().cpu(),
        "query_dim": controller.head.query_dim,
        "memory_dim": controller.head.memory_dim,
        "hidden_dim": controller.head.hidden_dim,
        "metadata": metadata or {},
    }
    torch.save(payload, Path(path))


def load_utility_head(path: Path, device="cpu") -> UtilityController:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("kind") != "utility_memory_v1":
        raise ValueError(f"Unsupported utility checkpoint: {payload.get('kind')!r}")
    head = UtilityHead(
        payload["query_dim"], payload["memory_dim"], payload["hidden_dim"]
    )
    head.load_state_dict(payload["head"])
    controller = UtilityController(head)
    controller.log_bias_scale.data.copy_(payload["log_bias_scale"])
    return controller.to(device)
