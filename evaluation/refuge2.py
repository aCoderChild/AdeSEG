"""REFUGE2 evaluation for two-region segmentation and structural measurements."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from adenoid.measurement import compute_ratio
from datasets.refuge2 import iter_refuge2_samples
from evaluation.ratio import compute_vcdr, evaluate_ratios
from evaluation.segmentation import calculate_scores


def _absolute_error(prediction: float, ground_truth: float) -> float:
    if not np.isfinite(prediction) or not np.isfinite(ground_truth):
        return float("nan")
    return float(abs(prediction - ground_truth))


def _correlation(rows, x_key: str, y_key: str) -> dict[str, float]:
    x = np.asarray([row[x_key] for row in rows], dtype=float)
    y = np.asarray([row[y_key] for row in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def _ratio_summary(rows, prediction_key: str, ground_truth_key: str, prefix: str) -> dict[str, float | int]:
    metrics = evaluate_ratios(
        [row[prediction_key] for row in rows],
        [row[ground_truth_key] for row in rows],
    )
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def evaluate_refuge2(
    prediction_dir: str | Path,
    data_root: str | Path,
    split: str,
    output_csv: str | Path,
):
    rows = []
    for sample in iter_refuge2_samples(data_root, split):
        path = Path(prediction_dir) / f"{sample.metadata['image_id']}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction: {path}")

        labels = np.asarray(Image.open(path).convert("L"))
        predicted_disc = labels != 255
        predicted_cup = labels == 0
        predicted_rim = predicted_disc & ~predicted_cup

        gt_disc = sample.masks["disc"]
        gt_cup = sample.masks["cup"]
        gt_rim = gt_disc & ~gt_cup

        predicted = {"disc": predicted_disc, "cup": predicted_cup, "rim": predicted_rim}
        ground_truth = {"disc": gt_disc, "cup": gt_cup, "rim": gt_rim}

        row = {"image_id": sample.metadata["image_id"]}
        for name in ("disc", "cup", "rim"):
            scores = calculate_scores(predicted[name], ground_truth[name])
            row[f"{name}_dice"] = scores["dice"]
            row[f"{name}_iou"] = scores["iou"]

        row["pred_vcdr"] = compute_vcdr(predicted_cup, predicted_disc)
        row["gt_vcdr"] = compute_vcdr(gt_cup, gt_disc)
        row["vcdr_abs_error"] = _absolute_error(row["pred_vcdr"], row["gt_vcdr"])

        row["pred_cup_rim_ratio"] = compute_ratio(
            predicted_cup, predicted_rim, mode="region_a_over_region_b"
        )
        row["gt_cup_rim_ratio"] = compute_ratio(
            gt_cup, gt_rim, mode="region_a_over_region_b"
        )
        row["cup_rim_abs_error"] = _absolute_error(
            row["pred_cup_rim_ratio"], row["gt_cup_rim_ratio"]
        )

        row["pred_cup_fraction"] = compute_ratio(
            predicted_cup, predicted_rim, mode="fraction_of_total"
        )
        row["gt_cup_fraction"] = compute_ratio(
            gt_cup, gt_rim, mode="fraction_of_total"
        )
        row["cup_fraction_abs_error"] = _absolute_error(
            row["pred_cup_fraction"], row["gt_cup_fraction"]
        )
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No REFUGE2 samples found for split {split!r}.")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, float | int] = {"count": len(rows)}
    for name in ("disc", "cup", "rim"):
        summary[f"{name}_dice"] = float(np.mean([row[f"{name}_dice"] for row in rows]))
        summary[f"{name}_iou"] = float(np.mean([row[f"{name}_iou"] for row in rows]))

    summary.update(_ratio_summary(rows, "pred_vcdr", "gt_vcdr", "vcdr"))
    summary.update(
        _ratio_summary(
            rows,
            "pred_cup_rim_ratio",
            "gt_cup_rim_ratio",
            "cup_rim_ratio",
        )
    )
    summary.update(
        _ratio_summary(
            rows,
            "pred_cup_fraction",
            "gt_cup_fraction",
            "cup_fraction",
        )
    )

    for ratio_name, error_key in (
        ("cup_rim_ratio", "cup_rim_abs_error"),
        ("cup_fraction", "cup_fraction_abs_error"),
    ):
        for region in ("cup", "rim"):
            correlation = _correlation(rows, f"{region}_dice", error_key)
            summary[f"{ratio_name}_error_vs_{region}_dice_pearson"] = correlation["pearson"]
            summary[f"{ratio_name}_error_vs_{region}_dice_spearman"] = correlation["spearman"]

    return summary
