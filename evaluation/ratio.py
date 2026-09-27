"""Evaluation of structural or clinical ratios against ground truth."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


def evaluate_ratios(
    predicted_ratios: Sequence[float], ground_truth_ratios: Sequence[float]
) -> dict[str, float | int]:
    """Return ratio MAE, RMSE, Pearson, and Spearman on finite paired values."""
    prediction = np.asarray(predicted_ratios, dtype=float)
    ground_truth = np.asarray(ground_truth_ratios, dtype=float)
    if prediction.shape != ground_truth.shape:
        raise ValueError("Predicted and ground-truth ratios must have the same shape.")
    valid = np.isfinite(prediction) & np.isfinite(ground_truth)
    prediction = prediction[valid]
    ground_truth = ground_truth[valid]
    if not prediction.size:
        return {"count": 0, "mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "spearman": float("nan")}

    error = prediction - ground_truth
    metrics: dict[str, float | int] = {
        "count": int(prediction.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson": float("nan"),
        "spearman": float("nan"),
    }
    if prediction.size >= 2 and np.std(prediction) > 0 and np.std(ground_truth) > 0:
        metrics["pearson"] = float(pearsonr(prediction, ground_truth).statistic)
        metrics["spearman"] = float(spearmanr(prediction, ground_truth).statistic)
    return metrics


def vertical_diameter(mask: np.ndarray) -> float:
    """Return the vertical extent of a non-empty binary mask in pixels."""
    rows = np.where(np.asarray(mask) > 0)[0]
    return float(rows.max() - rows.min() + 1) if rows.size else float("nan")


def compute_vcdr(cup_mask: np.ndarray, disc_mask: np.ndarray) -> float:
    """Compute vertical cup-to-disc ratio, returning nan for an empty disc."""
    cup, disc = vertical_diameter(cup_mask), vertical_diameter(disc_mask)
    return cup / disc if np.isfinite(cup) and np.isfinite(disc) and disc > 0 else float("nan")
