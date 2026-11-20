"""Budget-preserving, CMSA-inspired early-FPN enrichment of Lean-SAM2 memory.

This is a separately versioned experimental adapter, NOT the historical RMFA
logit refiner. The pretrained encoder runs first; its [B,Cm,Hm,Wm] memory is
then enriched with aligned early FPN features. Output has identical shape,
dtype and spatial coordinates. This module never creates extra bank tokens.

Untrained zero-residual identity is an engineering control, not an efficacy
claim. Eval integration requires a provenance-checked trained checkpoint, or
an explicitly named identity-control session.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryFeatureFusionConfig:
    early_channels: tuple[int, ...] = (256, 256)
    memory_channels: int = 64
    hidden_channels: int = 64
    max_relative_residual: float = 0.25
    preserve_native_no_object: bool = True

    def __post_init__(self):
        if (
            not isinstance(self.early_channels, tuple)
            or not 1 <= len(self.early_channels) <= 3
            or any(type(v) is not int or v <= 0 for v in self.early_channels)
            or type(self.memory_channels) is not int or self.memory_channels <= 0
            or type(self.hidden_channels) is not int or self.hidden_channels <= 0
            or type(self.max_relative_residual) is not float
            or not 0.0 < self.max_relative_residual <= 1.0
            or type(self.preserve_native_no_object) is not bool
        ):
            raise ValueError('Invalid memory feature fusion configuration')


def implementation_sha256() -> str:
    """Hash both the trainable module and its runtime adapter (source-bound)."""
    own = Path(__file__).resolve()
    hook = own.parents[1] / 'adeseg' / 'inference' / 'lean_memory_feature_fusion.py'
    h = sha256()
    for path in (own, hook):
        if not path.is_file():
            raise FileNotFoundError(f'Missing memory fusion implementation: {path}')
        data = path.read_bytes()
        h.update(path.name.encode('utf8') + b'\0' + len(data).to_bytes(8, 'big') + data)
    return h.hexdigest()


class EarlyFeatureMemoryFusion(nn.Module):
    """Learned, region-aware residual into the native 64-channel memory space.

    Unlike CMSA-Net's temporal cross-attention, this is a lightweight
    architectural hypothesis: align earlier SAM2 FPN stages to the already
    mask-conditioned memory output, add a bounded residual, retain native
    positional encodings and let Lean TAMP perform its unchanged pruning.
    """

    CHECKPOINT_FORMAT = 'adeseg_lean_early_memory_fusion_v1'

    def __init__(self, config: MemoryFeatureFusionConfig | None = None):
        super().__init__()
        self.config = config or MemoryFeatureFusionConfig()
        c = self.config
        self.early_projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, c.hidden_channels, 1, bias=False), nn.GELU())
            for ch in c.early_channels
        ])
        self.memory_projection = nn.Conv2d(c.memory_channels, c.hidden_channels, 1)
        # The predicted mask is an input feature, never a ground-truth input.
        mixed_channels = (len(c.early_channels) + 1) * c.hidden_channels + 1
        self.mixer = nn.Sequential(
            nn.Conv2d(mixed_channels, c.hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(c.hidden_channels, c.hidden_channels, 3,
                      padding=1, groups=c.hidden_channels),
            nn.GELU(),
        )
        self.output_projection = nn.Conv2d(c.hidden_channels, c.memory_channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self._trained_checkpoint_loaded = False

    def is_identity_initialized(self) -> bool:
        """No trainable residual can leak into a zero-initialized smoke test."""
        return bool(torch.count_nonzero(self.output_projection.weight).item() == 0
                    and torch.count_nonzero(self.output_projection.bias).item() == 0)

    def forward(self, early: Sequence[torch.Tensor], native_memory: torch.Tensor,
                predicted_mask_logits: torch.Tensor,
                object_score_logits: torch.Tensor) -> torch.Tensor:
        c = self.config
        if (not isinstance(early, (tuple, list))
                or len(early) != len(self.early_projections)):
            raise ValueError('Expected configured early FPN feature levels')
        if (not isinstance(native_memory, torch.Tensor) or native_memory.ndim != 4
                or native_memory.shape[1] != c.memory_channels
                or native_memory.shape[-2] < 1 or native_memory.shape[-1] < 1
                or not torch.is_floating_point(native_memory)):
            raise ValueError('Invalid native memory geometry')
        batch, _, height, width = native_memory.shape
        if (not isinstance(predicted_mask_logits, torch.Tensor)
                or predicted_mask_logits.ndim != 4
                or predicted_mask_logits.shape[:2] != (batch, 1)
                or not torch.is_floating_point(predicted_mask_logits)):
            raise ValueError('Expected one native predicted-mask logit map per object')
        if (not isinstance(object_score_logits, torch.Tensor)
                or object_score_logits.numel() != batch
                or not torch.is_floating_point(object_score_logits)):
            raise ValueError('Expected native object-presence logit for each object')
        model_device = self.output_projection.weight.device
        if (native_memory.device != model_device
                or predicted_mask_logits.device != model_device
                or object_score_logits.device != model_device):
            raise ValueError('Memory fusion features and parameters must share a device')
        for level, feature in enumerate(early):
            if (not isinstance(feature, torch.Tensor) or feature.ndim != 4
                    or feature.shape[0] != batch
                    or feature.shape[1] != c.early_channels[level]
                    or feature.shape[-2] < height or feature.shape[-1] < width
                    or not torch.is_floating_point(feature)
                    or feature.device != model_device):
                raise ValueError(f'Invalid early FPN level {level} geometry or device')
        all_inputs = [native_memory, predicted_mask_logits, object_score_logits, *early]
        if not all(bool(torch.isfinite(x).all().item()) for x in all_inputs):
            raise ValueError('Non-finite memory-fusion inputs')

        # Cast to the separately learned adapter precision, then restore
        # original memory precision exactly on return (mixed precision safe).
        compute_type = self.output_projection.weight.dtype
        target_hw = (height, width)
        projections = [
            layer(F.interpolate(feat, size=target_hw, mode='area')
                  .to(dtype=compute_type))
            for layer, feat in zip(self.early_projections, early)
        ]
        base = native_memory.to(dtype=compute_type)
        projections.append(self.memory_projection(base))
        mask = torch.sigmoid(predicted_mask_logits.float())
        mask = F.interpolate(mask, size=target_hw, mode='area').to(dtype=compute_type)
        signal = self.mixer(torch.cat((*projections, mask), dim=1))
        residual = self.output_projection(signal)

        # Relative bound is per memory entry. This limits amplification and
        # gives a well-defined ablation against output-only mask refinement.
        scale = base.detach().float().std(dim=(1, 2, 3), unbiased=False,
                                          keepdim=True).clamp_min(1e-6)
        residual = (c.max_relative_residual * scale.to(compute_type)) * torch.tanh(residual)
        # The conservative default preserves the native no-object memory
        # embedding. Disable this only as a separately named ablation.
        if c.preserve_native_no_object:
            present = (object_score_logits.reshape(batch, -1).float()[:, 0] > 0)
            residual = residual * present.to(dtype=compute_type).reshape(batch, 1, 1, 1)
        enriched = (base + residual).to(dtype=native_memory.dtype)
        if enriched.shape != native_memory.shape or not torch.isfinite(enriched).all():
            raise RuntimeError('Memory fusion violated the native geometry/finiteness contract')
        return enriched

    def load_trained_state(self, payload: dict, *, backbone_sha256: str,
                           train_manifest_sha256: str) -> None:
        """Load a source-, model-, dataset- and config-bound trained state.

        The caller must authenticate input checkpoint bytes separately and
        must never read training labels during inference.
        """
        self._trained_checkpoint_loaded = False
        if (not isinstance(payload, dict)
                or payload.get('format') != self.CHECKPOINT_FORMAT
                or payload.get('config') != asdict(self.config)
                or payload.get('implementation_sha256') != implementation_sha256()
                or payload.get('backbone_sha256') != backbone_sha256
                or payload.get('train_manifest_sha256') != train_manifest_sha256
                or type(payload.get('training_steps')) is not int
                or payload['training_steps'] <= 0):
            raise ValueError('Memory fusion checkpoint provenance mismatch')
        provided = payload.get('model_state_dict')
        expected = self.state_dict()
        if (not isinstance(provided, dict) or set(provided) != set(expected)
                or any(not isinstance(provided[k], torch.Tensor)
                       or provided[k].shape != expected[k].shape
                       or not torch.is_floating_point(provided[k])
                       or not bool(torch.isfinite(provided[k]).all().item())
                       for k in expected)):
            raise ValueError('Invalid trained memory fusion tensors')
        self.load_state_dict(provided, strict=True)
        self._trained_checkpoint_loaded = True


__all__ = ['MemoryFeatureFusionConfig', 'EarlyFeatureMemoryFusion', 'implementation_sha256']
