"""Validate archived automatic masks for a one-shot SAMURAI comparison."""
import numpy as np
from modeling.seq23_viewport import IMAGE_SIZE, VIEWPORT


def prepare_prompt(mask, sequence, expected_shape):
    """Return a Boolean prompt and its original-coordinate viewport.

    The input is an already verified automatic mask, never a label. Seq23
    accepts only the full-frame mask and the previously fixed main viewport.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(expected_shape):
        raise ValueError('Prompt and original image geometry differ')
    if not np.isin(mask, (0, 1, 255)).all():
        raise ValueError('Prompt must be a binary mask')
    mask = mask > 0
    if sequence == 'seq23':
        if tuple(expected_shape) != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
            raise ValueError('Seq23 original geometry differs')
        x0, y0, x1, y1 = VIEWPORT
        outside = mask.copy(); outside[y0:y1, x0:x1] = False
        if outside.any():
            raise ValueError('Seq23 prompt contains foreground outside the viewport')
        return mask[y0:y1, x0:x1].copy(), VIEWPORT
    return mask.copy(), (0, 0, expected_shape[1], expected_shape[0])
