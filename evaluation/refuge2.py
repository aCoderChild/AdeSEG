"""Disk-only REFUGE2 evaluation: disc/cup Dice and vertical CDR MAE."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.refuge2 import iter_refuge2_samples
from evaluation.ratio import compute_vcdr, evaluate_ratios
from evaluation.segmentation import calculate_scores


def evaluate_refuge2(prediction_dir: str | Path, data_root: str | Path, split: str, output_csv: str | Path):
    rows = []
    for sample in iter_refuge2_samples(data_root, split):
        path = Path(prediction_dir) / f"{sample.metadata['image_id']}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction: {path}")
        labels = np.asarray(Image.open(path).convert("L"))
        prediction = {"disc": labels != 255, "cup": labels == 0}
        row = {"image_id": sample.metadata["image_id"]}
        for name in ("disc", "cup"):
            scores = calculate_scores(prediction[name], sample.masks[name])
            row[f"{name}_dice"] = scores["dice"]
        row["pred_vcdr"] = compute_vcdr(prediction["cup"], prediction["disc"])
        row["gt_vcdr"] = compute_vcdr(sample.masks["cup"], sample.masks["disc"])
        rows.append(row)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    ratios = evaluate_ratios([row["pred_vcdr"] for row in rows], [row["gt_vcdr"] for row in rows])
    return {"count": len(rows), "disc_dice": float(np.mean([r["disc_dice"] for r in rows])), "cup_dice": float(np.mean([r["cup_dice"] for r in rows])), "vcdr_mae": ratios["mae"]}
