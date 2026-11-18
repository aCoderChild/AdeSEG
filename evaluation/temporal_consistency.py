"""Frame-adjacent temporal mask measures with explicit undefined cases.

Metrics are descriptive; raw mask change may reflect camera/object motion and
must NOT be interpreted automatically as tracking error.
"""
from __future__ import annotations
import numpy as np

NAMES = ("temporal_iou", "area_change_relative", "centroid_shift_normalized")

def adjacent_mask_metrics(previous, current):
    """Compare aligned HxW predicted masks; missing frames are handled by caller.

    Two empty masks: temporal IoU=1 and area change=0; centroid displacement
    is undefined (NaN). For transitions involving an empty mask, centroid shift
    remains undefined. Area difference is divided by max(previous area,
    current area, 1), returning [0,1].
    """
    a, b = np.asarray(previous), np.asarray(current)
    if a.ndim != 2 or a.shape != b.shape or not a.size:
        raise ValueError("Temporal masks must be nonempty aligned HxW arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Temporal mask values must be finite")
    a, b = a > 0, b > 0
    area_a, area_b = int(a.sum()), int(b.sum())
    union = int(np.logical_or(a, b).sum())
    intersection = int(np.logical_and(a, b).sum())
    tiou = float(intersection / union) if union else 1.0
    area_change = abs(area_b-area_a) / max(area_a, area_b, 1)
    shift = float("nan")
    if area_a and area_b:
        ya, xa = np.nonzero(a)
        yb, xb = np.nonzero(b)
        distance = np.hypot(float(xa.mean()-xb.mean()), float(ya.mean()-yb.mean()))
        diagonal = np.hypot(a.shape[0], a.shape[1])
        shift = float(distance / diagonal)
    return {"temporal_iou": tiou, "area_change_relative": float(area_change),
            "centroid_shift_normalized": shift}

def undefined_temporal_metrics():
    """First frame or missing adjacent prediction: do not invent an empty mask."""
    return {name: float("nan") for name in NAMES}
