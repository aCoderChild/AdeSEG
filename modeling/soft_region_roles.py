"""Continuous foreground, boundary and background memberships for memory tokens."""

import torch
import torch.nn.functional as F


def predicted_soft_regions(mask_logits):
    """Map stored 256x256 prediction logits to 64x64 soft role weights.

    The three weights are probabilities that two independent Bernoulli draws
    agree on background, agree on foreground, or disagree within a token cell.
    """
    logits = torch.as_tensor(mask_logits).detach().float()
    while logits.ndim > 2 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.shape != (256, 256) or not torch.isfinite(logits).all():
        raise ValueError("Expected finite native 256x256 stored mask logits")

    probability = F.avg_pool2d(logits.sigmoid()[None, None], 4, 4)[0, 0]
    background = (1 - probability).square()
    foreground = probability.square()
    boundary = 2 * probability * (1 - probability)
    memberships = torch.stack((background, foreground, boundary), dim=-1)
    if not torch.allclose(memberships.sum(dim=-1), torch.ones_like(probability), atol=1e-6):
        raise RuntimeError("Soft region memberships must sum to one per token")
    return memberships
