"""Per-region segmentation metrics for two-mask datasets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from utils.eval_metrics import calculate_scores


def evaluate_regions(
    predicted_masks: Mapping[str, np.ndarray], ground_truth_masks: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """Return named Dice and IoU values for matching region-mask mappings."""
    if set(predicted_masks) != set(ground_truth_masks):
        raise ValueError("Predicted and ground-truth region names must match.")
    metrics: dict[str, float] = {}
    for name in sorted(predicted_masks):
        scores = calculate_scores(predicted_masks[name], ground_truth_masks[name])
        metrics[f"{name}_dice"] = scores["dice"]
        metrics[f"{name}_iou"] = scores["iou"]
    return metrics
