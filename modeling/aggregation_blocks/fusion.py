"""Stages 4 and 6: upsample channel values and decode a mask-logit residual."""
import torch
from torch.nn import functional as F


class FeatureFusionBlock:
    """Residual fusion in image space at the native mask resolution."""

    @staticmethod
    def forward(fine_features, aggregated_values, *, output_projection):
        correction = F.interpolate(
            output_projection(aggregated_values),
            size=fine_features.shape[-2:], mode="bilinear", align_corners=False)
        return fine_features + correction


class ResidualMaskDecoderBlock:
    """Predict a logit correction while preserving the native SAM2 anchor."""

    @staticmethod
    def forward(fused_features, contrast, native_logits, *, mode, decoder_head):
        head_logits = native_logits.float()
        if mode == "foreground_background_asinh":
            head_logits = torch.asinh(head_logits)
        residual = decoder_head(
            torch.cat((fused_features, contrast, head_logits), dim=1))
        return native_logits + residual.to(dtype=native_logits.dtype)
