"""Change-supervised regional memory fusion adapted from MatAnyone."""

import torch
import torch.nn.functional as F
from torch import nn

from modeling.matanyone_temporal_uncertainty import build_temporal_uncertainty


class ChangeSupervisedRegionFusion(nn.Module):
    def __init__(self, channels=256, max_residual_rms_ratio=0.25):
        super().__init__()
        if channels <= 0 or not 0 < max_residual_rms_ratio <= 1:
            raise ValueError("Invalid channel count or residual trust-region ratio")
        self.uncertainty = build_temporal_uncertainty(channels, channels)
        self.regional_residual = nn.Conv2d(3 * channels, channels, 1, bias=False)
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.max_residual_rms_ratio = float(max_residual_rms_ratio)
        self.change_logits_history = []
        nn.init.zeros_(self.regional_residual.weight)
        with torch.no_grad():
            indices = torch.arange(channels)
            for offset in range(3):
                self.regional_residual.weight[indices, indices + offset * channels, 0, 0] = 1 / 3

    def forward(self, native, regional_reads, last_feature, current_feature,
                last_mask, memory_value_difference):
        expected = (native.shape[0], 3, native.shape[1], native.shape[2], native.shape[3])
        if native.ndim != 4 or regional_reads.shape != expected:
            raise ValueError("Expected native [B,C,H,W] and three aligned regional reads")
        for value in (last_feature, current_feature, memory_value_difference):
            if value.shape != native.shape:
                raise ValueError("Temporal feature inputs must match native read shape")
        if last_mask.shape != (native.shape[0], 1, native.shape[2], native.shape[3]):
            raise ValueError("Previous predicted mask must align with the query grid")

        dtype = native.dtype
        native, regional_reads, last_feature, current_feature, last_mask, memory_value_difference = (
            value.float() for value in
            (native, regional_reads, last_feature, current_feature, last_mask, memory_value_difference)
        )
        change_logits = self.uncertainty(
            last_feature, current_feature, last_mask, memory_value_difference
        )
        if self.training:
            self.change_logits_history.append(change_logits)
        change_probability = torch.sigmoid(change_logits)

        regional_memory = self.regional_residual(regional_reads.flatten(1, 2))
        # MatAnyone's memory/previous-frame fusion, applied to SAM2's regional readout.
        candidate = change_probability * regional_memory + (1 - change_probability) * last_feature
        delta = candidate - last_feature
        recent_rms = last_feature.square().mean(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        bounded_delta = torch.tanh(delta / recent_rms) * (
            self.max_residual_rms_ratio * recent_rms
        )
        output = native + torch.tanh(self.mix_logit) * bounded_delta
        if self.mix_logit.detach().item() == 0:
            output = output - output.detach() + native.detach()
        return output.to(dtype), change_probability


class ChangeSupervisedRegionHeads(nn.Module):
    def __init__(self, channels=256, layers=4):
        super().__init__()
        self.layers = nn.ModuleList(
            ChangeSupervisedRegionFusion(channels) for _ in range(layers)
        )


def temporal_change_loss(heads, samples, prompt_target):
    if not samples:
        raise ValueError("At least one predicted-memory training sample is required")
    if not torch.isfinite(prompt_target).all():
        raise ValueError("Prompt target must be finite")
    per_sample = []
    for index, sample in enumerate(samples):
        previous = prompt_target if index == 0 else samples[index - 1]["target"]
        current = sample["target"]
        previous = previous.to(device=current.device, dtype=current.dtype)
        layer_losses = []
        for head in heads.layers:
            if len(head.change_logits_history) != len(samples):
                raise RuntimeError("Uncertainty predictions do not cover every clip frame")
            logits = head.change_logits_history[index]
            target = (F.interpolate((current - previous).abs(), size=logits.shape[-2:],
                                    mode="area") > 0).float()
            layer_target = target.to(device=logits.device, dtype=logits.dtype)
            layer_losses.append(F.binary_cross_entropy_with_logits(logits, layer_target))
        per_sample.append(torch.stack(layer_losses).mean())
    return torch.stack(per_sample).mean()
