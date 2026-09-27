"""Archived learnable update for the compact-memory ablation."""

import torch
from torch import nn


class AdaptiveStateFusion(nn.Module):
    """Learn a spatial update gate while preserving native memory features."""

    def __init__(self, feature_channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden_channels = hidden_channels or feature_channels
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels * 3 + 1, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, feature_channels, 1),
            nn.Sigmoid(),
        )
        # At step zero this is exactly the fixed EMA alpha=0.1 baseline.
        # Keep the first layer initialized so the gate can learn from its inputs;
        # only the output layer must start at zero for an exact EMA update.
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)
        nn.init.constant_(self.gate[-2].bias, torch.logit(torch.tensor(0.1)).item())
        self.last_gate = None

    def forward(self, previous: torch.Tensor, candidate: torch.Tensor, foreground_probability: torch.Tensor) -> torch.Tensor:
        if previous.shape != candidate.shape or previous.ndim != 4:
            raise ValueError("previous and candidate must be equal [B, C, H, W] tensors.")
        probability = foreground_probability.to(device=previous.device, dtype=previous.dtype)
        if probability.shape != (previous.shape[0], 1, previous.shape[2], previous.shape[3]):
            raise ValueError("foreground_probability must be [B, 1, H, W] aligned to the state.")
        gate = self.gate(torch.cat((previous, candidate, (previous - candidate).abs(), probability), dim=1))
        self.last_gate = gate.detach()
        return (1.0 - gate) * previous + gate * candidate
