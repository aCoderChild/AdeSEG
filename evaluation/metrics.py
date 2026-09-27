"""Discrete binary and semantic multi-class segmentation metrics for MedSAM2."""

from collections.abc import Sequence

import numpy as np


_EPS = np.finfo(np.float64).eps


def _label_masks(prediction: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return identically shaped discrete label masks."""
    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction and ground-truth shapes differ: {prediction.shape} and {ground_truth.shape}."
        )
    return prediction, ground_truth


def _ratio(numerator: int, denominator: int, empty_value: float = 1.0) -> float:
    """Return the specified value for a zero denominator."""
    return float(numerator / denominator) if denominator else empty_value


def _binary_scores(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Compute scores for two boolean foreground masks."""
    true_positive = int(np.count_nonzero(prediction & ground_truth))
    false_positive = int(np.count_nonzero(prediction & ~ground_truth))
    false_negative = int(np.count_nonzero(~prediction & ground_truth))
    true_negative = int(np.count_nonzero(~prediction & ~ground_truth))
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return {
        "dice": _ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        "iou": _ratio(true_positive, true_positive + false_positive + false_negative),
        "f_measure": _ratio(2 * precision * recall, precision + recall, empty_value=0.0),
        "f2": 5 * precision * recall / (4 * precision + recall + _EPS),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": _ratio(true_negative, true_negative + false_positive),
        "accuracy": _ratio(true_positive + true_negative, prediction.size),
        "mae": float(np.mean(prediction != ground_truth)),
    }


def _multiclass_scores(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    class_labels: Sequence[int],
) -> dict[str, float]:
    """Macro-average foreground one-vs-rest scores for semantic classes."""
    labels = tuple(class_labels)
    if not labels:
        raise ValueError("class_labels must contain at least one foreground label.")
    if 0 in labels:
        raise ValueError("class_labels must not include background label 0.")
    if len(set(labels)) != len(labels):
        raise ValueError("class_labels must not contain duplicate labels.")

    scores_by_label = []
    for label in labels:
        predicted_label = prediction == label
        ground_truth_label = ground_truth == label
        # A class absent from both maps has no overlap to evaluate. Excluding it
        # avoids rewarding true-negative background pixels in a macro Dice/IoU.
        if not (predicted_label.any() or ground_truth_label.any()):
            continue
        scores_by_label.append(_binary_scores(predicted_label, ground_truth_label))

    metrics = _binary_scores(prediction == 0, ground_truth == 0)
    if scores_by_label:
        metrics.update({
            metric: float(np.mean([scores[metric] for scores in scores_by_label]))
            for metric in scores_by_label[0]
        })
    else:
        metrics.update({
            metric: float("nan")
            for metric in metrics
            if metric not in {"accuracy", "mae"}
        })
    # Accuracy and MAE are label-map metrics, not one-vs-rest macro metrics.
    metrics["accuracy"] = float(np.mean(prediction == ground_truth))
    metrics["mae"] = float(np.mean(prediction != ground_truth))
    return metrics


def calculate_scores(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    class_labels: Sequence[int] | None = None,
) -> dict[str, float]:
    """Compute binary or semantic multi-class segmentation scores.

    By default, every nonzero pixel is binary foreground. This is the
    appropriate PolypGen behavior because prediction label ``1`` and
    ground-truth label ``255`` denote the same polyp class. To score semantic
    classes, supply their nonzero IDs with ``class_labels``. Foreground classes
    are evaluated one-vs-rest and macro-averaged; background is excluded.
    """
    prediction, ground_truth = _label_masks(prediction, ground_truth)
    if class_labels is None:
        return _binary_scores(prediction > 0, ground_truth > 0)
    return _multiclass_scores(prediction, ground_truth, class_labels)
