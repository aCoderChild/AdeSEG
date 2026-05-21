#!/usr/bin/env python3
"""Aggregate MedSAM2 polypgen sequence metrics into summary CSV files.

This script scans:
  external/MedSAM2/data/polypgen_vid_seq/seq{number}/outputs/metrics.csv

It writes two outputs into scripts/analysis:
  - per_image_metrics.csv
  - seq_average_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]

# Base directories containing per-sequence folders (seq1, seq2, ...)
DEFAULT_MEDSAM2_BASE = Path("external/MedSAM2/data/polypgen_vid_seq")
DEFAULT_YOLO_BASE = Path("external/YOLO_SAM2/positive_cropped")

# Defaults for output analysis folders
DEFAULT_ANALYSIS_MEDSAM2 = Path("scripts/analysis/MedSAM2")
DEFAULT_ANALYSIS_MEDSAM2_FIRST = Path("scripts/analysis/MedSAM2_first_frame")
DEFAULT_ANALYSIS_YOLO = Path("scripts/analysis/YOLO_SAM2")

PER_IMAGE_OUTPUT = "per_image_metrics.csv"
SEQ_AVERAGE_OUTPUT = "seq_average_metrics.csv"


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def iter_sequence_dirs(base_dir: Path) -> Iterable[Path]:
    for seq_dir in sorted(base_dir.glob("seq*")):
        if seq_dir.is_dir():
            yield seq_dir


def seq_sort_key(seq_name: str) -> tuple[int, str]:
    prefix = "seq"
    if seq_name.startswith(prefix):
        try:
            return int(seq_name[len(prefix) :]), seq_name
        except ValueError:
            pass
    return (10**9, seq_name)


def image_sort_key(image_name: str) -> tuple[int, str]:
    stem = Path(image_name).stem
    try:
        return int(stem), image_name
    except ValueError:
        return (10**9, image_name)


def load_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"))
    return mask > 0


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    if mask.size == 0 or not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    return float(np.mean(ys)), float(np.mean(xs))


def get_prediction_mask_dir(seq_dir: Path, subfolder: str) -> Path:
    if subfolder == "script_gen":
        return seq_dir / subfolder / "predicted_mask"
    return seq_dir / subfolder / "images"


def normalize_csv_row(row: dict[str, str]) -> dict[str, str]:
    return {
        (key or "").strip(): (value or "").strip()
        for key, value in row.items()
        if key is not None
    }


def read_metrics_csv(metrics_path: Path) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    per_image_rows: list[dict[str, str]] = []
    average_rows: list[dict[str, str]] = []

    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized_row = normalize_csv_row(row)
            # Detect per-image rows by presence of image_name or image/frame column
            if "image_name" in normalized_row or "image" in normalized_row:
                per_image_rows.append(normalized_row)
            else:
                average_rows.append(normalized_row)

    # Return per-image rows and a consolidated average dict (if any)
    average_row = average_rows[0] if average_rows else None
    return per_image_rows, average_row


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(normalize_csv_row(row))
    return rows


def split_metrics_rows(metrics_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    per_image_rows: list[dict[str, str]] = []
    average_rows: list[dict[str, str]] = []

    for row in metrics_rows:
        frame_name = row.get("image_name", "").strip()
        if frame_name.lower() == "average":
            average_rows.append(row)
        else:
            per_image_rows.append(row)

    return per_image_rows, average_rows


def normalize_metric_key(row: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key, "").strip()
        if value != "":
            return value
    return default


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate per-sequence metrics from multiple pipelines into analysis CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_MEDSAM2_BASE,
        help="Base directory containing seq* for MedSAM2 (contains 'outputs' and 'outputs_first_frame')",
    )
    parser.add_argument(
        "--medsam2-output-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_MEDSAM2,
        help="Directory where aggregated MedSAM2 CSV files will be written",
    )
    parser.add_argument(
        "--medsam2-first-output-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_MEDSAM2_FIRST,
        help="Directory where aggregated MedSAM2 first-frame CSV files will be written",
    )
    parser.add_argument(
        "--yolo-base",
        type=Path,
        default=DEFAULT_YOLO_BASE,
        help="Base directory containing seq* for YOLO_SAM2",
    )
    parser.add_argument(
        "--yolo-output-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_YOLO,
        help="Directory where aggregated YOLO_SAM2 CSV files will be written",
    )
    args = parser.parse_args()
    med_base = resolve_repo_path(args.input_dir)
    med_out = resolve_repo_path(args.medsam2_output_dir)
    med_first_out = resolve_repo_path(args.medsam2_first_output_dir)
    yolo_base = resolve_repo_path(args.yolo_base)
    yolo_out = resolve_repo_path(args.yolo_output_dir)

    def aggregate_pipeline(base_dir: Path, subfolder: str, out_dir: Path) -> None:
        per_image_rows_all: list[dict[str, str]] = []
        seq_average_rows_all: list[dict[str, str]] = []

        for seq_dir in iter_sequence_dirs(base_dir):
            seq_name = seq_dir.name
            per_image_path = seq_dir / subfolder / PER_IMAGE_OUTPUT
            seq_average_path = seq_dir / subfolder / SEQ_AVERAGE_OUTPUT
            metrics_path = seq_dir / subfolder / "metrics.csv"

            if per_image_path.exists() and seq_average_path.exists():
                per_image = read_csv_rows(per_image_path)
                seq_avg = read_csv_rows(seq_average_path)
            elif metrics_path.exists():
                metrics_rows = read_csv_rows(metrics_path)
                per_image, seq_avg = split_metrics_rows(metrics_rows)
            else:
                print(
                    f"Skipping {seq_name}: missing {per_image_path}, {seq_average_path}, and {metrics_path}"
                )
                continue

            for row in per_image:
                per_image_rows_all.append(
                    {
                        "seq": seq_name,
                        "frame_id": row.get("frame_id") or row.get("image_name") or row.get("image") or row.get("frame") or "",
                        "dice": row.get("dice", ""),
                        "iou": row.get("iou", ""),
                        "recall": row.get("recall", ""),
                        "precision": row.get("precision", ""),
                        "f_score": row.get("f_score", "") or row.get("f_score", ""),
                    }
                )

            # Take the first seq_average row if multiple
            if seq_avg:
                avg = seq_avg[0]
                seq_average_rows_all.append(
                    {
                        "seq": seq_name,
                        "mean_dice": normalize_metric_key(avg, "mean_dice", "dice"),
                        "mean_iou": normalize_metric_key(avg, "mean_iou", "iou"),
                        "mean_recall": normalize_metric_key(avg, "mean_recall", "recall"),
                        "mean_precision": normalize_metric_key(avg, "mean_precision", "precision"),
                        "mean_f_score": normalize_metric_key(avg, "mean_f_score", "f_score"),
                        "ciou": normalize_metric_key(avg, "ciou", "mean_ciou"),
                        "centroid_disp_mean": normalize_metric_key(avg, "centroid_disp_mean"),
                        "pred_area_std_norm": normalize_metric_key(avg, "pred_area_std_norm"),
                        "missing_pred_count": 0,
                    }
                )

        per_image_rows_all.sort(key=lambda row: (seq_sort_key(row["seq"]), image_sort_key(row["frame_id"])))
        seq_average_rows_all.sort(key=lambda row: seq_sort_key(row["seq"]))

        out_dir = resolve_repo_path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Compute mask-derived metrics for each sequence.
        seq_to_total_frames: dict[str, int] = {}
        seq_to_missing_pred_count: dict[str, int] = {}
        seq_to_pred_areas: dict[str, list[float]] = {}
        seq_to_centroids: dict[str, list[tuple[float, float] | None]] = {}
        seq_to_pred_masks: dict[str, list[np.ndarray]] = {}

        rows_by_seq: dict[str, list[dict[str, str]]] = {}
        for row in per_image_rows_all:
            rows_by_seq.setdefault(row["seq"], []).append(row)

        for seq_dir in iter_sequence_dirs(base_dir):
            seq_name = seq_dir.name
            mask_dir = get_prediction_mask_dir(seq_dir, subfolder)
            seq_rows = sorted(rows_by_seq.get(seq_name, []), key=lambda row: image_sort_key(row["frame_id"]))
            seq_to_total_frames[seq_name] = len(seq_rows)
            seq_to_missing_pred_count[seq_name] = 0
            seq_to_pred_areas[seq_name] = []
            seq_to_centroids[seq_name] = []
            seq_to_pred_masks[seq_name] = []

            missing_count = 0
            for row in seq_rows:
                frame_id = row["frame_id"]
                mask_path = mask_dir / f"{Path(frame_id).stem}.png"
                if not mask_path.exists():
                    missing_count += 1
                    continue

                mask = load_binary_mask(mask_path)
                seq_to_pred_masks[seq_name].append(mask)
                seq_to_pred_areas[seq_name].append(float(mask.sum()))
                seq_to_centroids[seq_name].append(mask_centroid(mask))

            seq_to_missing_pred_count[seq_name] = missing_count

        for row in seq_average_rows_all:
            seq_name = row["seq"]
            pred_areas = seq_to_pred_areas.get(seq_name, [])
            centroids = seq_to_centroids.get(seq_name, [])
            pred_masks = seq_to_pred_masks.get(seq_name, [])

            ciou_values: list[float] = []
            disp_values: list[float] = []
            for idx in range(len(pred_masks) - 1):
                pred_a = pred_masks[idx]
                pred_b = pred_masks[idx + 1]
                inter = int((pred_a & pred_b).sum())
                union = int((pred_a | pred_b).sum())
                ciou_values.append(float(inter / union) if union > 0 else 0.0)

                c1 = centroids[idx]
                c2 = centroids[idx + 1]
                if c1 is not None and c2 is not None:
                    dy = c2[0] - c1[0]
                    dx = c2[1] - c1[1]
                    disp_values.append(float(np.hypot(dy, dx)))

            pred_area_mean = float(np.mean(pred_areas)) if pred_areas else 0.0
            pred_area_std = float(np.std(pred_areas)) if pred_areas else 0.0
            row["missing_pred_count"] = seq_to_missing_pred_count.get(seq_name, 0)
            row["ciou"] = float(np.mean(ciou_values)) if ciou_values else 0.0
            row["centroid_disp_mean"] = float(np.mean(disp_values)) if disp_values else 0.0
            row["pred_area_std_norm"] = float(pred_area_std / pred_area_mean) if pred_area_mean > 0 else 0.0

        write_csv(
            out_dir / PER_IMAGE_OUTPUT,
            ["seq", "frame_id", "dice", "iou", "recall", "precision", "f_score"],
            per_image_rows_all,
        )
        write_csv(
            out_dir / SEQ_AVERAGE_OUTPUT,
            [
                "seq",
                "mean_dice",
                "mean_iou",
                "mean_recall",
                "mean_precision",
                "mean_f_score",
                "ciou",
                "centroid_disp_mean",
                "pred_area_std_norm",
                "missing_pred_count",
            ],
            seq_average_rows_all,
        )
        print(f"Wrote {len(per_image_rows_all)} per-image rows to {out_dir / PER_IMAGE_OUTPUT}")
        print(f"Wrote {len(seq_average_rows_all)} sequence-average rows to {out_dir / SEQ_AVERAGE_OUTPUT}")

    # Aggregate MedSAM2 (standard outputs)
    aggregate_pipeline(med_base, "outputs", med_out)
    # Aggregate MedSAM2 first-frame outputs
    aggregate_pipeline(med_base, "outputs_first_frame", med_first_out)
    # Aggregate YOLO_SAM2 script_gen outputs
    aggregate_pipeline(yolo_base, "script_gen", yolo_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
