"""Robust aggregation of valid frame-level measurements."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .frame_selection import select_valid_frames


def aggregate_video_ratio(
    frame_predictions: Iterable[dict[str, object]], method: str = "median"
) -> float:
    """Aggregate valid frame ratios into one video-level score."""
    valid_frames = select_valid_frames(frame_predictions)
    if not valid_frames:
        return float("nan")
    ratios = np.asarray([frame["ratio"] for frame in valid_frames], dtype=float)
    if method == "median":
        return float(np.median(ratios))
    if method == "mean":
        return float(np.mean(ratios))
    raise ValueError("method must be 'median' or 'mean'.")
