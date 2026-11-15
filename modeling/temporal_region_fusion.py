"""Uncertainty-gated residual fusion of three predicted-mask memory reads."""

import torch
from torch import nn

from modeling.matanyone_temporal_uncertainty import build_temporal_uncertainty


class TemporalRegionFusion(nn.Module):
    def __init__(self, channels=256, max_residual_rms_ratio=0.25):
        super().__init__()
        if channels <= 0 or not 0 < max_residual_rms_ratio <= 1:
            raise ValueError("Invalid channel count or residual trust-region ratio")
        self.uncertainty = build_temporal_uncertainty(channels, channels)
        self.regional_residual = nn.Conv2d(3 * channels, channels, 1, bias=False)
        self.max_residual_rms_ratio = float(max_residual_rms_ratio)
        nn.init.zeros_(self.regional_residual.weight)

    def forward(self, native, regional_reads, last_feature, current_feature,
                last_mask, memory_value_difference):
        if native.ndim != 4 or regional_reads.shape != (
            native.shape[0], 3, native.shape[1], native.shape[2], native.shape[3]
        ):
            raise ValueError("Expected native [B,C,H,W] and three aligned regional reads")
        for value in (last_feature, current_feature, memory_value_difference):
            if value.shape != native.shape:
                raise ValueError("Temporal feature inputs must match native read shape")
        if last_mask.shape != (native.shape[0], 1, native.shape[2], native.shape[3]):
            raise ValueError("Previous predicted mask must align with the query grid")
        if not all(torch.isfinite(x).all() for x in
                   (native, regional_reads, last_feature, current_feature,
                    last_mask, memory_value_difference)):
            raise ValueError("Fusion inputs must be finite")

        dtype = native.dtype
        native = native.float()
        regional_reads = regional_reads.float()
        last_feature = last_feature.float()
        current_feature = current_feature.float()
        last_mask = last_mask.float()
        memory_value_difference = memory_value_difference.float()
        logits = self.uncertainty(last_feature, current_feature, last_mask,
                                  memory_value_difference)
        gate = torch.sigmoid(logits)
        residual = self.regional_residual(regional_reads.flatten(1, 2))
        native_rms = native.square().mean(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        residual = residual.tanh() * (self.max_residual_rms_ratio * native_rms)
        correction = gate * residual
        if torch.count_nonzero(self.regional_residual.weight.detach()).item() == 0:
            correction = correction - correction.detach()
        return (native + correction).to(dtype), gate
