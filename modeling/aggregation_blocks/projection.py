"""Stage 1: align early image features and deep memory-conditioned features."""
import torch
from torch.nn import functional as F


class FeatureProjectionBlock:
    """Project each scale to C/2 channels, then concatenate at mask resolution."""

    @staticmethod
    def forward(early, deep, *, early_projection, deep_projection, output_size, mask_only=False):
        if mask_only:
            early, deep = torch.zeros_like(early), torch.zeros_like(deep)
        early_map = F.interpolate(
            early_projection(early.float()), size=output_size,
            mode="bilinear", align_corners=False)
        deep_map = F.interpolate(
            deep_projection(deep.float()), size=output_size,
            mode="bilinear", align_corners=False)
        return torch.cat((early_map, deep_map), dim=1)


class SpatialContextBlock:
    """Stage 2: construct compact features and a soft mask-derived region prior."""

    @staticmethod
    def forward(fine_features, native_logits, *, attention_size):
        size = native_logits.shape[-2:]
        coarse = (min(size[0], attention_size), min(size[1], attention_size))
        context = F.adaptive_avg_pool2d(fine_features, coarse)
        foreground_probability = F.interpolate(
            torch.sigmoid(native_logits.float()), size=coarse, mode="area")
        return context, foreground_probability
