"""Per-frame two-region measurements with a configurable ratio definition."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


RATIO_MODES = ("fraction_of_total", "region_a_over_region_b")


def masks_are_valid(*masks: np.ndarray, min_area: int = 1) -> bool:
    """Check that all masks are 2-D, shape-compatible, and visible."""
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


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got {mask.shape}.")
    return mask > 0


def mask_area(mask: np.ndarray) -> int:
    """Count foreground pixels in one binary or label mask."""
    return int(np.count_nonzero(_binary_mask(mask)))


def compute_ratio(
    region_a_mask: np.ndarray,
    region_b_mask: np.ndarray,
    mode: str,
) -> float:
    """Compute a two-region area ratio without fixing a clinical definition.

    ``fraction_of_total`` returns ``area_a / (area_a + area_b)`` and is bounded
    in [0, 1]. ``region_a_over_region_b`` returns ``area_a / area_b``. Empty
    denominators return ``nan`` so callers cannot silently treat them as valid.
    """
    region_a = _binary_mask(region_a_mask)
    region_b = _binary_mask(region_b_mask)
    if region_a.shape != region_b.shape:
        raise ValueError("Region masks must have the same shape.")
    if mode not in RATIO_MODES:
        raise ValueError(f"Unsupported ratio mode {mode!r}; choose from {RATIO_MODES}.")

    area_a = int(np.count_nonzero(region_a))
    area_b = int(np.count_nonzero(region_b))
    denominator = area_a + area_b if mode == "fraction_of_total" else area_b
    return float(area_a / denominator) if denominator else float("nan")


def measure_frame(
    frame_idx: int,
    region_a_mask: np.ndarray,
    region_b_mask: np.ndarray,
    mode: str,
    region_a_name: str = "adenoid",
    region_b_name: str = "airway",
    min_area: int = 1,
) -> dict[str, object]:
    """Create the common downstream prediction record for one two-mask frame."""
    if not region_a_name or not region_b_name or region_a_name == region_b_name:
        raise ValueError("Region names must be distinct non-empty strings.")
    region_a = _binary_mask(region_a_mask)
    region_b = _binary_mask(region_b_mask)
    ratio = compute_ratio(region_a, region_b, mode=mode)
    return {
        "frame_idx": int(frame_idx),
        f"{region_a_name}_mask": region_a,
        f"{region_b_name}_mask": region_b,
        f"{region_a_name}_area": mask_area(region_a),
        f"{region_b_name}_area": mask_area(region_b),
        "ratio": ratio,
        "ratio_mode": mode,
        "valid_frame": masks_are_valid(region_a, region_b, min_area=min_area)
        and bool(np.isfinite(ratio)),
    }


def select_valid_frames(frame_predictions: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Keep frame records marked valid with a finite ratio."""
    return [
        prediction for prediction in frame_predictions
        if prediction.get("valid_frame") and np.isfinite(prediction.get("ratio", float("nan")))
    ]


def aggregate_video_ratio(frame_predictions: Iterable[dict[str, object]], method: str = "median") -> float:
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
