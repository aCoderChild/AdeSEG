"""Archived helpers to train AdaptiveStateFusion while keeping MedSAM2 frozen.

This module intentionally contains only the project-specific optimization logic.
Dataset sampling and MedSAM2 clip construction should reuse the existing
MedSAM2 training stack rather than duplicate it here.
"""

from __future__ import annotations

import torch

from training.losses import dice_bce_loss
from training.video_trainer import freeze_except_fusion


def training_step(model, predicted_logits, target_masks):
    """Return the fusion loss for a propagated clip."""
    freeze_except_fusion(model)
    return dice_bce_loss(predicted_logits, target_masks)


def build_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """Optimize only the adaptive fusion parameters."""
    fusion = freeze_except_fusion(model)
    return torch.optim.AdamW(fusion.parameters(), lr=lr, weight_decay=weight_decay)
