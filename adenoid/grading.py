"""Configurable conversion from a continuous ratio to an ordinal grade."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

import numpy as np


def ratio_to_grade(ratio: float, thresholds: Sequence[float]) -> int:
    """Map a finite ratio to an ordinal grade using ascending lower boundaries.

    For thresholds ``[0.25, 0.5]``, ratios below 0.25 map to grade 0, ratios in
    [0.25, 0.5) map to grade 1, and ratios at least 0.5 map to grade 2.
    Clinical thresholds must be supplied by the validated annotation protocol.
    """
    ratio = float(ratio)
    thresholds = tuple(float(value) for value in thresholds)
    if not np.isfinite(ratio):
        raise ValueError("ratio must be finite.")
    if any(not np.isfinite(value) for value in thresholds):
        raise ValueError("thresholds must be finite.")
    if tuple(sorted(thresholds)) != thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("thresholds must be strictly ascending.")
    return bisect_right(thresholds, ratio)
