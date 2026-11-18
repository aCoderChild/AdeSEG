"""Training-only adaptation of CMSA-Net's structure objective.

The author code at ``external/CMSA-Net/cmsa_project/scripts/my_train.py``
uses a 31-pixel local average to emphasize boundaries in BCE and IoU. Its
``reduce='none'`` argument is a legacy boolean argument and produces a mean
scalar in the current PyTorch; this adaptation deliberately uses per-pixel
BCE. The Dice term is the existing project soft-Dice formula, not the
author's separate implementation. No labels or loss are used at inference.
"""

import torch
import torch.nn.functional as F


def cmsa_boundary_loss(prediction, target):
    """Corrected weighted BCE and aggregated IoU plus project soft Dice.

    We use two separable average pools rather than a 31x31 pool. They are
    equivalent to the author's zero-padded box average, aside from floating
    point roundoff, but avoid an expensive full-resolution 31x31 operation.
    The pooling window remains 31 native-resolution pixels: this is an
    exploratory choice and differs in relative scale from 352-pixel inputs.
    """
    if prediction.ndim != 4 or prediction.shape[1] != 1 or prediction.shape != target.shape:
        raise ValueError('Prediction and target must be matching N x 1 x H x W tensors')
    if min(prediction.shape) <= 0 or min(prediction.shape[-2:]) < 16:
        raise ValueError('Prediction dimensions are too small for the pooling window')
    if prediction.device != target.device or not all(torch.isfinite(t).all() for t in (prediction, target)):
        raise ValueError('Prediction and target must be finite and on the same device')
    if bool((target < 0).any()) or bool((target > 1).any()):
        raise ValueError('Target must be between zero and one')
    target = target.detach().float()
    prediction = prediction.float()
    local = F.avg_pool2d(target, (1, 31), stride=1, padding=(0, 15))
    local = F.avg_pool2d(local, (31, 1), stride=1, padding=(15, 0))
    weight = 1 + 5 * (local - target).abs()
    dims = (-2, -1)
    weighted_bce = (weight * F.binary_cross_entropy_with_logits(prediction, target, reduction='none')).sum(dims) / weight.sum(dims)
    probability = prediction.sigmoid()
    intersection = (weight * probability * target).sum(dims)
    union = (weight * (probability + target - probability * target)).sum(dims)
    weighted_iou = 1 - (intersection + 1) / (union + 1)
    dice = 1 - (2 * (probability * target).sum(dims) + 1) / (probability.sum(dims) + target.sum(dims) + 1)
    return (weighted_bce + weighted_iou + dice).mean()
