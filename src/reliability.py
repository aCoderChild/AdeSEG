from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveReliabilityEstimator(nn.Module):
    """Simple MedSAM-compatible reliability estimator returning r_t in [0, 1]."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim * 3 + 1, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        frame_features: torch.Tensor,
        previous_memory: torch.Tensor | None,
        candidate_memory: torch.Tensor,
        mask_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        previous = torch.zeros_like(candidate_memory)
        if previous_memory is not None:
            previous = F.interpolate(
                previous_memory,
                size=candidate_memory.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        mask = torch.zeros(
            candidate_memory.size(0),
            1,
            *candidate_memory.shape[-2:],
            device=candidate_memory.device,
            dtype=candidate_memory.dtype,
        )
        if mask_logits is not None:
            mask = torch.sigmoid(mask_logits)
            mask = F.interpolate(
                mask,
                size=candidate_memory.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([frame_features, previous, candidate_memory, mask], dim=1)
        return torch.sigmoid(self.net(x))


class QDMNScoreReliabilityEstimator(nn.Module):
    """
    Reliability estimator backed by QDMN's ``Score`` head.

    QDMN's original score head expects 1024-channel, 24x24 features. This adapter
    projects MedSAM features to that shape, applies QDMN's head, then squashes the
    output to a probability.
    """

    def __init__(
        self,
        feature_dim: int,
        qdmn_root: str | Path = "external/QDMN",
        checkpoint_path: str | Path | None = None,
        qdmn_channels: int = 1024,
        score_input_size: int = 24,
    ) -> None:
        super().__init__()
        qdmn_root = Path(qdmn_root).resolve()
        if str(qdmn_root) not in sys.path:
            sys.path.insert(0, str(qdmn_root))

        try:
            _install_numpy_arraysetops_shim()
            from model.modules import Score
        except Exception as exc:  # pragma: no cover - depends on local QDMN deps
            raise ImportError(f"Could not import QDMN Score from {qdmn_root}") from exc

        self.project = nn.Conv2d(feature_dim * 3 + 1, qdmn_channels, 1)
        self.score = Score(qdmn_channels)
        self.score_input_size = score_input_size
        if checkpoint_path is not None:
            self.load_qdmn_score_weights(checkpoint_path)

    def load_qdmn_score_weights(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _unwrap_state_dict(checkpoint)
        score_state = {}
        for key, value in state_dict.items():
            clean_key = key.removeprefix("module.")
            if clean_key.startswith("score."):
                score_state[clean_key.removeprefix("score.")] = value

        if not score_state:
            raise RuntimeError(
                f"No QDMN score.* weights found in checkpoint: {checkpoint_path}"
            )

        missing, unexpected = self.score.load_state_dict(score_state, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected QDMN score checkpoint keys: {unexpected}"
            )
        if missing:
            print(
                "Warning: QDMN score checkpoint is missing keys "
                f"{missing}; continuing with initialized values for them."
            )

    def forward(
        self,
        frame_features: torch.Tensor,
        previous_memory: torch.Tensor | None,
        candidate_memory: torch.Tensor,
        mask_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        previous = torch.zeros_like(candidate_memory)
        if previous_memory is not None:
            previous = F.interpolate(
                previous_memory,
                size=candidate_memory.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        mask = torch.zeros(
            candidate_memory.size(0),
            1,
            *candidate_memory.shape[-2:],
            device=candidate_memory.device,
            dtype=candidate_memory.dtype,
        )
        if mask_logits is not None:
            mask = torch.sigmoid(mask_logits)
            mask = F.interpolate(
                mask,
                size=candidate_memory.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([frame_features, previous, candidate_memory, mask], dim=1)
        x = self.project(x)
        x = F.interpolate(
            x,
            size=(self.score_input_size, self.score_input_size),
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(self.score(x))


def _install_numpy_arraysetops_shim() -> None:
    """Keep QDMN importable with newer NumPy versions."""

    module_name = "numpy.lib.arraysetops"
    if module_name in sys.modules:
        return

    shim = types.ModuleType(module_name)
    shim.isin = np.isin
    sys.modules[module_name] = shim


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "state_dict", "network", "module"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint
