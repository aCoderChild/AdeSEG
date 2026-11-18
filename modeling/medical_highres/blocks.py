"""Parameter-free named stages of MedSAM2 high-resolution refinement.

Parameters remain directly registered on the adapter, preserving the saved
five-layer checkpoint's original state_dict key names.
"""
import torch
from torch.nn import functional as F


class CrossScaleProjection:
    """Project Hiera-T early (32-channel) and deep (256-channel) features."""

    @staticmethod
    def forward(early, deep, *, early_proj, deep_proj):
        early_map = early_proj(early.float())
        deep_map = F.interpolate(
            deep_proj(deep.float()), size=early.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        return torch.cat((early_map, deep_map), dim=1)


class DepthwiseSpatialFusion:
    """Fuse projected scales, then refine local texture and boundaries."""

    @staticmethod
    def forward(projected, *, fuse, spatial):
        features = F.gelu(fuse(projected))
        return F.gelu(spatial(features))


class BoundedSkipResidual:
    """Bound a learned high-resolution skip correction and retain native skip."""

    @staticmethod
    def forward(native_skip, features, *, output, maximum_delta):
        correction = maximum_delta * torch.tanh(output(features))
        return native_skip + correction.to(native_skip.dtype)
