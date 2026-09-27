"""Prompt helpers shared by manual and ground-truth target workflows."""

from __future__ import annotations

import numpy as np


def box_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Return the tight xyxy box of a nonempty mask, or ``None`` when empty."""
    foreground = np.asarray(mask) > 0
    if foreground.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got {foreground.shape}.")
    if not foreground.any():
        return None
    y, x = np.where(foreground)
    return np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float32)
