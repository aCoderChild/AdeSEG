"""Basic mask-level validity checks shared by quantitative pipelines."""

from __future__ import annotations

import numpy as np


def masks_are_valid(*masks: np.ndarray, min_area: int = 1) -> bool:
    """Return whether every mask is 2-D, shape-compatible, and sufficiently visible.

    Blur, occlusion, and anatomy-specific plausibility require a validated
    clinical protocol, so they are deliberately not inferred here.
    """
    if min_area < 1:
        raise ValueError("min_area must be at least 1.")
    if not masks:
        return False
    arrays = [np.asarray(mask) for mask in masks]
    if any(mask.ndim != 2 for mask in arrays):
        return False
    if any(mask.shape != arrays[0].shape for mask in arrays[1:]):
        return False
    return all(np.count_nonzero(mask) >= min_area for mask in arrays)
