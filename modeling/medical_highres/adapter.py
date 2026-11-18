"""Reconstructed MedSAM2 high-resolution skip adapter.

Parameter names match saved 2026-11-17 checkpoints, but the historical
forward equation is absent from GitHub. Successful strict weight loading
must NOT be presented as numerical reproduction of the saved experiment.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn

from .blocks import CrossScaleProjection, DepthwiseSpatialFusion, BoundedSkipResidual


class MedicalHighResolutionAdapter(nn.Module):
    """Early/deep feature fusion with a bounded decoder high-res skip residual.

    Default dimensions reproduce the five-layer saved checkpoint's tensor
    shapes. Residual magnitude is a NEW declared reconstruction setting.
    """

    KEY_NAMES = (
        "early_proj.weight", "early_proj.bias", "deep_proj.weight", "deep_proj.bias",
        "fuse.weight", "fuse.bias", "spatial.weight", "spatial.bias",
        "output.weight", "output.bias",
    )

    def __init__(self, *, early_channels=32, deep_channels=256, skip_channels=32,
                 maximum_delta=0.1):
        super().__init__()
        if any(not isinstance(x, int) or x < 1 for x in (
                early_channels, deep_channels, skip_channels)):
            raise ValueError("Channels must be positive integers")
        if not 0.0 < maximum_delta <= 1.0:
            raise ValueError("maximum_delta must be in (0, 1]")
        self.early_channels = early_channels
        self.deep_channels = deep_channels
        self.skip_channels = skip_channels
        self.maximum_delta = maximum_delta
        self.early_proj = nn.Conv2d(early_channels, skip_channels, 1)
        self.deep_proj = nn.Conv2d(deep_channels, skip_channels, 1)
        self.fuse = nn.Conv2d(skip_channels * 2, skip_channels, 1)
        self.spatial = nn.Conv2d(skip_channels, skip_channels, 3,
                                 padding=1, groups=skip_channels)
        self.output = nn.Conv2d(skip_channels, skip_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.checkpoint_sha256 = None
        self.checkpoint_format = None

    def forward(self, early, deep, native_skip):
        if (early.ndim != 4 or deep.ndim != 4 or native_skip.ndim != 4
                or early.shape[0] != deep.shape[0]
                or early.shape[0] != native_skip.shape[0]
                or early.shape[1] != self.early_channels
                or deep.shape[1] != self.deep_channels
                or native_skip.shape[1] != self.skip_channels
                or early.shape[-2:] != native_skip.shape[-2:]):
            raise ValueError("MedSAM2 early/deep/skip feature dimensions mismatch")
        if len({early.device, deep.device, native_skip.device}) != 1:
            raise ValueError("MedSAM2 features must be on the same device")
        if not all(torch.isfinite(x).all().item() for x in (early, deep, native_skip)):
            raise ValueError("Nonfinite MedSAM2 feature input")
        projected = CrossScaleProjection.forward(
            early, deep, early_proj=self.early_proj, deep_proj=self.deep_proj)
        fused = DepthwiseSpatialFusion.forward(
            projected, fuse=self.fuse, spatial=self.spatial)
        return BoundedSkipResidual.forward(
            native_skip, fused, output=self.output,
            maximum_delta=self.maximum_delta)

    def load_historical_state(self, path, *, expected_sha256, allowed_formats=None):
        """Strict weight loading; historical output/forward parity is unverified."""
        path = Path(path).expanduser().resolve(strict=True)
        if (not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or any(x not in "0123456789abcdef" for x in expected_sha256)):
            raise ValueError("Expected lowercase complete SHA256 is required")
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("Historical adapter checkpoint SHA256 mismatch")
        record = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(record, dict) or not isinstance(record.get("state_dict"), dict):
            raise ValueError("Missing historical state_dict")
        if allowed_formats is not None and record.get("format") not in allowed_formats:
            raise ValueError("Unexpected historical adapter checkpoint format")
        state = record["state_dict"]
        if set(state) != set(self.KEY_NAMES):
            raise ValueError("Checkpoint module names do not match historical adapter")
        if any(not isinstance(w, torch.Tensor) or not bool(torch.isfinite(w).all())
               for w in state.values()):
            raise ValueError("Nonfinite adapter checkpoint weights")
        self.load_state_dict(state, strict=True)
        self.checkpoint_sha256 = actual_sha256
        self.checkpoint_format = record.get("format")
        return record
