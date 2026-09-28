"""Project-specific segmentation losses used by RGM training."""

import torch
import torch.nn.functional as F


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary BCE + soft Dice loss for propagated segmentation masks."""
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + (1.0 - dice.mean())
