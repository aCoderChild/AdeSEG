#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def load_mask(path: Path) -> np.ndarray:
    mask = Image.open(path).convert("L")
    arr = np.array(mask)
    return arr > 0


def safe_div(numer: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return float(numer) / float(denom)


def compute_frame_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    dice = safe_div(2 * inter, pred_sum + gt_sum)
    iou = safe_div(inter, union)
    precision = safe_div(inter, pred_sum)
    recall = safe_div(inter, gt_sum)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pred_area": float(pred_sum),
        "gt_area": float(gt_sum),
    }


def centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    if mask.sum() == 0:
        return None
    ys, xs = np.nonzero(mask)
    return float(np.mean(ys)), float(np.mean(xs))


def compute_consecutive_iou(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    inter = int((pred_a & pred_b).sum())
    union = int((pred_a | pred_b).sum())
    return safe_div(inter, union)


def std_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(values))


def mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def list_masks(path: Path) -> Dict[str, Path]:
    masks: Dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in path.glob(ext):
            if p.is_file():
                masks[p.stem] = p
    return masks


def evaluate_sequence(seq_name: str, pred_dir: Path, gt_dir: Path) -> Tuple[List[Dict], Dict]:
    pred_files = list_masks(pred_dir)
    gt_files = list_masks(gt_dir)

    common = sorted(set(pred_files.keys()) & set(gt_files.keys()))
    missing_pred = sorted(set(gt_files.keys()) - set(pred_files.keys()))
    missing_gt = sorted(set(pred_files.keys()) - set(gt_files.keys()))

    frame_rows: List[Dict] = []
    pred_masks: List[np.ndarray] = []
    centroids: List[Optional[Tuple[float, float]]] = []
    areas: List[float] = []

    for name in common:
        pred = load_mask(pred_files[name])
        gt = load_mask(gt_files[name])
        metrics = compute_frame_metrics(pred, gt)
        metrics.update({"seq": seq_name, "frame": name})
        frame_rows.append(metrics)
        pred_masks.append(pred)
        centroids.append(centroid(pred))
        areas.append(metrics["pred_area"])

    ciou_values: List[float] = []
    disp_values: List[float] = []
    for i in range(len(pred_masks) - 1):
        ciou_values.append(compute_consecutive_iou(pred_masks[i], pred_masks[i + 1]))
        c1 = centroids[i]
        c2 = centroids[i + 1]
        if c1 is not None and c2 is not None:
            dy = c2[0] - c1[0]
            dx = c2[1] - c1[1]
            disp_values.append(float(np.hypot(dy, dx)))

    seq_metrics = {
        "seq": seq_name,
        "frame_count": len(common),
        "mean_dice": mean_or_zero([r["dice"] for r in frame_rows]),
        "mean_iou": mean_or_zero([r["iou"] for r in frame_rows]),
        "mean_precision": mean_or_zero([r["precision"] for r in frame_rows]),
        "mean_recall": mean_or_zero([r["recall"] for r in frame_rows]),
        "mean_ciou": mean_or_zero(ciou_values),
        "pred_area_mean": mean_or_zero(areas),
        "pred_area_std": std_or_zero(areas),
        "pred_area_std_norm": safe_div(std_or_zero(areas), mean_or_zero(areas)),
        "centroid_disp_mean": mean_or_zero(disp_values),
        "centroid_disp_std": std_or_zero(disp_values),
        "missing_pred_count": len(missing_pred),
        "missing_gt_count": len(missing_gt),
    }

    return frame_rows, seq_metrics


def evaluate_pipeline(pipeline_name: str, base_dir: Path, pred_subpath: str, gt_subpath: str) -> Tuple[List[Dict], List[Dict]]:
    frame_rows: List[Dict] = []
    seq_rows: List[Dict] = []

    for seq_dir in sorted([d for d in base_dir.iterdir() if d.is_dir()]):
        pred_dir = seq_dir / pred_subpath
        gt_dir = seq_dir / gt_subpath
        if not pred_dir.exists() or not gt_dir.exists():
            continue
        seq_name = seq_dir.name
        seq_frame_rows, seq_metrics = evaluate_sequence(seq_name, pred_dir, gt_dir)
        for row in seq_frame_rows:
            row["pipeline"] = pipeline_name
            frame_rows.append(row)
        seq_metrics["pipeline"] = pipeline_name
        seq_rows.append(seq_metrics)

    return frame_rows, seq_rows


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    (key or "").strip(): (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
            )
        return rows


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def load_analysis_pipeline(pipeline_name: str, pipeline_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    per_image_path = pipeline_dir / "per_image_metrics.csv"
    seq_average_path = pipeline_dir / "seq_average_metrics.csv"
    if not per_image_path.exists():
        raise FileNotFoundError(f"Missing per-image metrics file: {per_image_path}")
    if not seq_average_path.exists():
        raise FileNotFoundError(f"Missing sequence-average metrics file: {seq_average_path}")

    per_image_source_rows = read_csv_rows(per_image_path)
    seq_average_source_rows = read_csv_rows(seq_average_path)

    frame_rows: List[Dict] = []
    seq_rows: List[Dict] = []

    frame_count_by_seq: Dict[str, int] = {}
    for row in per_image_source_rows:
        seq_name = str(row.get("seq", "")).strip()
        frame_count_by_seq[seq_name] = frame_count_by_seq.get(seq_name, 0) + 1
        frame_rows.append(
            {
                "pipeline": pipeline_name,
                "seq": seq_name,
                "frame": str(row.get("frame_id", row.get("image_name", ""))).strip(),
                "dice": to_float(row.get("dice")),
                "iou": to_float(row.get("iou")),
                "precision": to_float(row.get("precision")),
                "recall": to_float(row.get("recall")),
                "pred_area": 0.0,
                "gt_area": 0.0,
            }
        )

    for row in seq_average_source_rows:
        seq_name = str(row.get("seq", "")).strip()
        seq_rows.append(
            {
                "pipeline": pipeline_name,
                "seq": seq_name,
                "frame_count": frame_count_by_seq.get(seq_name, 0),
                "mean_dice": to_float(row.get("mean_dice", row.get("dice"))),
                "mean_iou": to_float(row.get("mean_iou", row.get("iou"))),
                "mean_precision": to_float(row.get("mean_precision", row.get("precision"))),
                "mean_recall": to_float(row.get("mean_recall", row.get("recall"))),
                "mean_ciou": to_float(row.get("ciou", row.get("mean_ciou"))),
                "pred_area_mean": to_float(row.get("pred_area_mean")),
                "pred_area_std": to_float(row.get("pred_area_std")),
                "pred_area_std_norm": to_float(row.get("pred_area_std_norm")),
                "centroid_disp_mean": to_float(row.get("centroid_disp_mean")),
                "centroid_disp_std": to_float(row.get("centroid_disp_std")),
                "missing_pred_count": int(to_float(row.get("missing_pred_count"))),
                "missing_gt_count": 0,
            }
        )

    return frame_rows, seq_rows


def summarize(seq_rows: List[Dict]) -> Dict[str, float]:
    if not seq_rows:
        return {
            "seq_count": 0,
            "mean_dice": 0.0,
            "mean_iou": 0.0,
            "mean_ciou": 0.0,
            "centroid_disp_mean": 0.0,
            "pred_area_std_norm": 0.0,
        }

    return {
        "seq_count": len(seq_rows),
        "mean_dice": mean_or_zero([r["mean_dice"] for r in seq_rows]),
        "mean_iou": mean_or_zero([r["mean_iou"] for r in seq_rows]),
        "mean_ciou": mean_or_zero([r["mean_ciou"] for r in seq_rows]),
        "centroid_disp_mean": mean_or_zero([r["centroid_disp_mean"] for r in seq_rows]),
        "pred_area_std_norm": mean_or_zero([r["pred_area_std_norm"] for r in seq_rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate analysis CSVs for MedSAM2, MedSAM2_1_frame_bbox, and YOLO_SAM2")
    parser.add_argument(
        "--medsam2_dir",
        type=Path,
        default=Path("scripts/analysis/MedSAM2"),
        help="Directory containing MedSAM2 per_image_metrics.csv and seq_average_metrics.csv",
    )
    parser.add_argument(
        "--medsam2_1_frame_bbox_dir",
        type=Path,
        default=Path("scripts/analysis/MedSAM2_first_frame"),
        help="Directory containing MedSAM2_first_frame per_image_metrics.csv and seq_average_metrics.csv",
    )
    parser.add_argument(
        "--yolo_sam2_dir",
        type=Path,
        default=Path("scripts/analysis/YOLO_SAM2"),
        help="Directory containing YOLO_SAM2 per_image_metrics.csv and seq_average_metrics.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("scripts/eval_outputs"),
    )
    args = parser.parse_args()

    med_frame, med_seq = load_analysis_pipeline("MedSAM2", args.medsam2_dir)
    med_1_frame_frame, med_1_frame_seq = load_analysis_pipeline(
        "MedSAM2_first_frame",
        args.medsam2_1_frame_bbox_dir,
    )
    yolo_frame, yolo_seq = load_analysis_pipeline("YOLO_SAM2", args.yolo_sam2_dir)

    frame_rows = med_frame + med_1_frame_frame + yolo_frame
    seq_rows = med_seq + med_1_frame_seq + yolo_seq

    frame_fields = [
        "pipeline",
        "seq",
        "frame",
        "dice",
        "iou",
        "precision",
        "recall",
        "pred_area",
        "gt_area",
    ]
    seq_fields = [
        "pipeline",
        "seq",
        "frame_count",
        "mean_dice",
        "mean_iou",
        "mean_precision",
        "mean_recall",
        "mean_ciou",
        "pred_area_mean",
        "pred_area_std",
        "pred_area_std_norm",
        "centroid_disp_mean",
        "centroid_disp_std",
        "missing_pred_count",
        "missing_gt_count",
    ]

    write_csv(args.out_dir / "frame_metrics.csv", frame_rows, frame_fields)
    write_csv(args.out_dir / "seq_metrics.csv", seq_rows, seq_fields)

    med_summary = summarize(med_seq)
    yolo_summary = summarize(yolo_seq)

    summary_rows = [
        {"pipeline": "MedSAM2", **med_summary},
        {"pipeline": "MedSAM2_first_frame", **summarize(med_1_frame_seq)},
        {"pipeline": "YOLO_SAM2", **yolo_summary},
    ]
    summary_fields = [
        "pipeline",
        "seq_count",
        "mean_dice",
        "mean_iou",
        "mean_ciou",
        "centroid_disp_mean",
        "pred_area_std_norm",
    ]
    write_csv(args.out_dir / "summary.csv", summary_rows, summary_fields)

    print(f"Wrote {len(frame_rows)} frame rows to {args.out_dir / 'frame_metrics.csv'}")
    print(f"Wrote {len(seq_rows)} sequence rows to {args.out_dir / 'seq_metrics.csv'}")
    print(f"Wrote summary to {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
