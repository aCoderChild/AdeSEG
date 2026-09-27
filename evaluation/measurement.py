"""Measurement-aware evaluation for a named pair of anatomical masks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from adenoid.measurement import compute_ratio
from evaluation.segmentation import evaluate_regions


def evaluate_two_region_measurement(
    predicted_masks: Mapping[str, np.ndarray],
    ground_truth_masks: Mapping[str, np.ndarray],
    region_a_name: str,
    region_b_name: str,
    ratio_mode: str,
) -> dict[str, float]:
    """Report per-region overlap and the error in an explicit ratio protocol.

    ``ratio_mode`` is required because the clinical obstruction definition must
    come from the annotated adenoid-data protocol rather than a code default.
    """
    if set(predicted_masks) != set(ground_truth_masks):
        raise ValueError("Predicted and ground-truth region names must match.")
    if region_a_name not in predicted_masks or region_b_name not in predicted_masks:
        raise ValueError("The requested measurement regions are missing.")

    metrics = evaluate_regions(predicted_masks, ground_truth_masks)
    predicted_ratio = compute_ratio(
        predicted_masks[region_a_name], predicted_masks[region_b_name], mode=ratio_mode
    )
    ground_truth_ratio = compute_ratio(
        ground_truth_masks[region_a_name], ground_truth_masks[region_b_name], mode=ratio_mode
    )
    metrics.update({
        "predicted_ratio": predicted_ratio,
        "ground_truth_ratio": ground_truth_ratio,
        "ratio_absolute_error": (
            float(abs(predicted_ratio - ground_truth_ratio))
            if np.isfinite(predicted_ratio) and np.isfinite(ground_truth_ratio)
            else float("nan")
        ),
    })
    return metrics
