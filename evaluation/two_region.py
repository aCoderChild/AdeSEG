"""Two-region evaluation of indexed prediction masks against a manifest.

Per frame: Dice/IoU of each region and the two-region area ratio for prediction
and ground truth (``fraction_of_total`` = A / (A + B), e.g. adenoid / (adenoid +
airway)). Summary: mean region Dice/IoU, ratio MAE/RMSE/correlations, and, when
grade thresholds are given, grade accuracy and Cohen's kappa. Videos with more
than one frame are also scored on their median ratio. Predictions are read from
``<pred_dir>/masks/<video_id>/<image stem>.png`` as written by scripts/infer.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adenoid.grading import ratio_to_grade
from adenoid.io import load_semantic_mask
from datasets.video import load_manifest
from evaluation.measurement import evaluate_two_region_measurement
from evaluation.ratio import evaluate_ratios


def cohen_kappa(a: list[int], b: list[int]) -> float:
    if not a:
        return float("nan")
    classes = sorted(set(a) | set(b))
    index = {c: i for i, c in enumerate(classes)}
    table = np.zeros((len(classes), len(classes)))
    for x, y in zip(a, b):
        table[index[x], index[y]] += 1
    n = table.sum()
    observed = np.trace(table) / n
    expected = float((table.sum(0) * table.sum(1)).sum()) / n ** 2
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def evaluate(manifest, split, pred_dir, region_a, region_b, ratio_mode, thresholds=None):
    (name_a, label_a), (name_b, label_b) = region_a, region_b
    rows, videos = [], load_manifest(manifest, split)
    for video_id, frames in videos.items():
        for frame in frames:
            truth = load_semantic_mask(frame.mask_path)
            prediction = np.asarray(Image.open(Path(pred_dir) / "masks" / video_id / f"{frame.image_path.stem}.png"))
            if prediction.shape != truth.shape:
                raise ValueError(f"Prediction and ground truth differ in size for {frame.image_path}")
            row = {"video_id": video_id, "frame_index": frame.frame_index, "frame": frame.image_path.stem}
            row.update(evaluate_two_region_measurement(
                {name_a: prediction == label_a, name_b: prediction == label_b},
                {name_a: truth == label_a, name_b: truth == label_b},
                name_a, name_b, ratio_mode,
            ))
            if thresholds and np.isfinite(row["predicted_ratio"]) and np.isfinite(row["ground_truth_ratio"]):
                row["predicted_grade"] = ratio_to_grade(row["predicted_ratio"], thresholds)
                row["ground_truth_grade"] = ratio_to_grade(row["ground_truth_ratio"], thresholds)
            rows.append(row)

    summary = {"split": split, "frames": len(rows), "videos": len(videos), "ratio_mode": ratio_mode,
               "regions": {name_a: label_a, name_b: label_b}}
    for name in (name_a, name_b):
        for metric in ("dice", "iou"):
            summary[f"{name}_{metric}"] = float(np.mean([row[f"{name}_{metric}"] for row in rows]))
    summary["frame_ratio"] = evaluate_ratios([r["predicted_ratio"] for r in rows], [r["ground_truth_ratio"] for r in rows])
    if thresholds:
        graded = [r for r in rows if "predicted_grade" in r]
        pred, true = [r["predicted_grade"] for r in graded], [r["ground_truth_grade"] for r in graded]
        summary["grade"] = {"thresholds": list(thresholds), "count": len(graded),
                            "accuracy": float(np.mean([p == t for p, t in zip(pred, true)])) if graded else float("nan"),
                            "kappa": cohen_kappa(true, pred)}
    if max(len(frames) for frames in videos.values()) > 1:
        median = lambda values: float(np.median(values)) if values else float("nan")
        per_video = {}
        for row in rows:
            per_video.setdefault(row["video_id"], []).append(row)
        summary["video_ratio"] = evaluate_ratios(
            [median([r["predicted_ratio"] for r in v if np.isfinite(r["predicted_ratio"])]) for v in per_video.values()],
            [median([r["ground_truth_ratio"] for r in v if np.isfinite(r["ground_truth_ratio"])]) for v in per_video.values()],
        )
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--pred_dir", type=Path, required=True, help="scripts/infer.py --output_dir")
    parser.add_argument("--region_a", required=True, metavar="NAME:LABEL", help="numerator region, e.g. adenoid:2")
    parser.add_argument("--region_b", required=True, metavar="NAME:LABEL", help="e.g. airway:1")
    parser.add_argument("--ratio_mode", required=True, choices=("fraction_of_total", "region_a_over_region_b"))
    parser.add_argument("--grade_thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--output_dir", type=Path, default=None, help="default: <pred_dir>/two_region")
    args = parser.parse_args()
    parse = lambda text: (text.split(":")[0], int(text.split(":")[1]))
    rows, summary = evaluate(args.manifest, args.split, args.pred_dir, parse(args.region_a), parse(args.region_b),
                             args.ratio_mode, args.grade_thresholds)
    output_dir = args.output_dir or args.pred_dir / "two_region"
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "per_frame.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
