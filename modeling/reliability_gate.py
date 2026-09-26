"""Learned reliability gate shared by training and inference."""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedReliabilityGate(nn.Module):
    """Predict the scalar recurrent-state update weight for each object."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, predicted_iou, object_score, state_similarity):
        signals = torch.stack(
            (predicted_iou, object_score, state_similarity), dim=-1
        )
        return self.network(signals).squeeze(-1)
