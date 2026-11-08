"""Expose SAM2 decoder masks for training before presence suppression."""
from contextlib import contextmanager

import torch


@contextmanager
def decoder_training_mask(decoder):
    captured = []
    hook = decoder.register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        yield captured
    finally:
        hook.remove()


def select_decoder_mask(captured):
    if len(captured) != 1:
        raise RuntimeError('Expected exactly one SAM2 decoder call')
    masks, ious = captured[0][:2]
    if masks.ndim != 4 or ious.shape != masks.shape[:2]:
        raise ValueError('Unexpected SAM2 mask and IoU shapes')
    index = ious.argmax(dim=1)
    return masks[torch.arange(masks.shape[0], device=masks.device), index].unsqueeze(1).float()


def present_mask_loss(mask_logits, target, loss_fn):
    """Apply mask supervision only to objects present in the target."""
    present = target.flatten(1).any(dim=1)
    if not present.any():
        return mask_logits.sum() * 0.0
    return loss_fn(mask_logits[present], target[present])
