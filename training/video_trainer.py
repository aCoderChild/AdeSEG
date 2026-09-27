"""Fusion-only training helpers for PolypGen and future adenoid video clips."""

import torch

from modeling.fusion import AdaptiveStateFusion
from training.losses import dice_bce_loss


def freeze_except_fusion(model):
    """Freeze MedSAM2 and leave only the adaptive state-fusion module trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    fusion = getattr(model, "state_fusion", None)
    if not isinstance(fusion, AdaptiveStateFusion):
        raise ValueError("Model must be built with state_update_mode='adaptive'.")
    for parameter in fusion.parameters():
        parameter.requires_grad = True
    return fusion


def fusion_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """Create an AdamW optimizer over the fusion module only."""
    fusion = freeze_except_fusion(model)
    return torch.optim.AdamW(fusion.parameters(), lr=lr, weight_decay=weight_decay)


__all__ = ["dice_bce_loss", "freeze_except_fusion", "fusion_optimizer"]
