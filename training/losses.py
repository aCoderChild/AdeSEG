"""Segmentation losses used by image and video trainers."""

import torch
from torch.nn import functional as F


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(dtype=logits.dtype)
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(-2, -1))
    dice = 1 - (2 * intersection + 1) / (probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + dice.mean()
