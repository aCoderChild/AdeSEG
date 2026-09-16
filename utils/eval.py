#!/usr/bin/env python3
"""Evaluate MedSAM2 mask outputs on PolypGen sequences.

MedSAM2 inference scripts write one indexed mask per frame to
``<output_mask_dir>/seqN/<frame>.png``. This evaluator compares those masks
with PolypGen masks and writes score tables plus visual overlays only.
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if __package__:
    from .eval_metrics import calculate_scores
    from .mask_utils import make_overlay, save_overlay
else:
    from eval_metrics import calculate_scores
    from mask_utils import make_overlay, save_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "data" / "PolypGen2021_MultiCenterData_v3" / "sequenceData" / "positive"
)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
METRIC_NAMES = (
    "dice", "iou", "f_measure", "f2", "precision", "recall", "sensitivity",
    "specificity", "accuracy", "mae", "temporal_iou",
)


def natural_sort_key(path: Path) -> list[object]:
    """Sort frame names numerically when their stems contain frame numbers."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def discover_sequences(data_root: Path) -> list[str]:
    """Return PolypGen sequence directory names, such as ``seq1``."""
    return sorted(
        (path.name for path in data_root.iterdir() if path.is_dir() and re.fullmatch(r"seq\d+", path.name)),
        key=lambda name: natural_sort_key(Path(name)),
    )


def sequence_directories(data_root: Path, sequence_name: str) -> tuple[Path, Path]:
    """Resolve the actual PolypGen image and mask directories for one sequence."""
    sequence_dir = data_root / sequence_name
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory does not exist: {sequence_dir}")
    sequence_number = sequence_name.removeprefix("seq")
    image_dir = sequence_dir / f"images_seq{sequence_number}"
    mask_dir = sequence_dir / f"masks_seq{sequence_number}"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"{sequence_name} must contain {image_dir.name} and {mask_dir.name}.")
    return image_dir, mask_dir


def image_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_sort_key,
    )


def load_label_mask(path: Path) -> np.ndarray:
    """Load a single-channel indexed mask without changing object labels."""
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        # This is only a grayscale-RGB fallback, not a colour-map conversion.
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got {mask.shape} from {path}")
    return mask


def load_polypgen_ground_truth_mask(path: Path) -> np.ndarray:
    """Load PolypGen GT, restoring binary JPEG masks after lossy compression."""
    mask = load_label_mask(path)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return (mask >= 128).astype(np.uint8)
    return mask


def frame_stem(mask_path: Path) -> str:
    """Map a common ``*_mask`` GT name to its endoscopy-frame stem."""
    stem = mask_path.stem
    return stem[:-5] if stem.endswith("_mask") else stem


def find_frame(directory: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def load_prediction_mask(prediction_dir: Path, stem: str, shape: tuple[int, int]) -> np.ndarray | None:
    """Load a direct MedSAM2 mask, or combine optional per-object mask folders."""
    direct_path = find_frame(prediction_dir, stem)
    if direct_path is not None:
        prediction = load_label_mask(direct_path)
    else:
        object_masks: list[tuple[int, np.ndarray]] = []
        for object_dir in sorted((path for path in prediction_dir.iterdir() if path.is_dir()), key=natural_sort_key):
            object_path = find_frame(object_dir, stem)
            if object_path is None:
                continue
            try:
                object_label = int(object_dir.name)
            except ValueError:
                object_label = len(object_masks) + 1
            object_masks.append((object_label, load_label_mask(object_path)))
        if not object_masks:
            return None
        prediction = np.zeros_like(object_masks[0][1], dtype=np.int32)
        for object_label, object_mask in object_masks:
            prediction[object_mask > 0] = object_label
    if prediction.shape != shape:
        prediction = cv2.resize(prediction, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return prediction


def segmentation_scores(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Delegate binary or indexed-mask scoring to the shared metric module."""
    return calculate_scores(prediction, ground_truth)


def temporal_iou(previous_prediction: np.ndarray | None, prediction: np.ndarray) -> float:
    """IoU of consecutive foreground predictions; this is a stability diagnostic."""
    if previous_prediction is None:
        return float("nan")
    return calculate_scores(prediction > 0, previous_prediction > 0)["iou"]


def create_overlay(frame_bgr: np.ndarray, prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Create the shared RGB prediction/ground-truth overlay."""
    if frame_bgr.shape[:2] != ground_truth.shape:
        frame_bgr = cv2.resize(frame_bgr, (ground_truth.shape[1], ground_truth.shape[0]))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return make_overlay(frame_rgb, ground_truth, prediction)


def mean_metric(rows: list[dict[str, object]], metric: str) -> float:
    values = np.asarray([row[metric] for row in rows], dtype=float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if values.size else float("nan")


def evaluate_sequence(
    data_root: Path,
    output_mask_dir: Path,
    evaluation_dir: Path,
    sequence_name: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate one sequence and save one overlay for each ground-truth frame."""
    image_dir, ground_truth_dir = sequence_directories(data_root, sequence_name)
    prediction_dir = output_mask_dir / sequence_name
    ground_truth_files = image_files(ground_truth_dir)
    if not ground_truth_files:
        raise FileNotFoundError(f"No ground-truth masks found in {ground_truth_dir}")

    overlay_dir = evaluation_dir / "overlays" / sequence_name
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    previous_prediction: np.ndarray | None = None
    missing_predictions = 0

    for ground_truth_path in ground_truth_files:
        stem = frame_stem(ground_truth_path)
        ground_truth = load_polypgen_ground_truth_mask(ground_truth_path)
        prediction = load_prediction_mask(prediction_dir, stem, ground_truth.shape) if prediction_dir.is_dir() else None
        prediction_missing = prediction is None
        if prediction_missing:
            prediction = np.zeros_like(ground_truth)
            missing_predictions += 1

        image_path = find_frame(image_dir, stem)
        if image_path is None:
            raise FileNotFoundError(f"No endoscopy image for {ground_truth_path.name} in {image_dir}")
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError(f"Could not read endoscopy image: {image_path}")
        overlay_path = overlay_dir / f"{stem}.png"
        save_overlay(create_overlay(frame_bgr, prediction, ground_truth), overlay_path)

        scores = segmentation_scores(prediction, ground_truth)
        scores["temporal_iou"] = temporal_iou(previous_prediction, prediction)
        rows.append({"sequence": sequence_name, "frame": stem, "prediction_missing": prediction_missing, **scores})
        previous_prediction = prediction

    sequence_row: dict[str, object] = {
        "sequence": sequence_name,
        "frames": len(rows),
        "missing_predictions": missing_predictions,
    }
    sequence_row.update({metric: mean_metric(rows, metric) for metric in METRIC_NAMES})
    return sequence_row, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_memory_confidence_statistics(
    output_mask_dir: Path,
    evaluation_dir: Path,
    sequence_names: list[str],
) -> None:
    """Summarize MedSAM2 predicted-IoU confidence records when inference saved them."""
    confidence_path = output_mask_dir / "memory_confidence_per_frame.csv"
    if not confidence_path.is_file():
        return

    with confidence_path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    non_conditioning = [
        record
        for record in records
        if record.get("output_type") == "non_cond_frame_outputs"
    ]
    scores_by_sequence: dict[str, list[float]] = {name: [] for name in sequence_names}
    for record in non_conditioning:
        try:
            scores_by_sequence.setdefault(record["sequence"], []).append(
                float(record["iou_prediction"])
            )
        except (KeyError, TypeError, ValueError):
            continue

    per_sequence_rows = []
    for sequence_name in sequence_names:
        values = np.asarray(scores_by_sequence.get(sequence_name, []), dtype=float)
        per_sequence_rows.append({
            "sequence": sequence_name,
            "non_conditioning_frames": int(values.size),
            "mean_iou_prediction": float(values.mean()) if values.size else float("nan"),
            "std_iou_prediction": float(values.std()) if values.size else float("nan"),
            "min_iou_prediction": float(values.min()) if values.size else float("nan"),
            "max_iou_prediction": float(values.max()) if values.size else float("nan"),
        })
    write_csv(
        evaluation_dir / "memory_confidence_per_sequence.csv",
        [
            "sequence", "non_conditioning_frames", "mean_iou_prediction",
            "std_iou_prediction", "min_iou_prediction", "max_iou_prediction",
        ],
        per_sequence_rows,
    )

    pooled_scores = np.asarray(
        [score for values in scores_by_sequence.values() for score in values], dtype=float
    )
    sequence_means = np.asarray(
        [row["mean_iou_prediction"] for row in per_sequence_rows], dtype=float
    )
    sequence_means = sequence_means[~np.isnan(sequence_means)]
    stats_rows = []
    for aggregation, values in (("frame", pooled_scores), ("sequence", sequence_means)):
        stats_rows.append({
            "aggregation": aggregation,
            "records": int(values.size),
            "mean_iou_prediction": float(values.mean()) if values.size else float("nan"),
            "std_iou_prediction": float(values.std()) if values.size else float("nan"),
            "min_iou_prediction": float(values.min()) if values.size else float("nan"),
            "max_iou_prediction": float(values.max()) if values.size else float("nan"),
        })
    write_csv(
        evaluation_dir / "memory_confidence_stats.csv",
        [
            "aggregation", "records", "mean_iou_prediction", "std_iou_prediction",
            "min_iou_prediction", "max_iou_prediction",
        ],
        stats_rows,
    )


def evaluate_masks(
    output_mask_dir: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
    evaluation_dir: Path | None = None,
    sequences: list[str] | None = None,
) -> dict[str, object]:
    """Evaluate MedSAM2 outputs and return aggregate frame-level scores."""
    output_mask_dir = output_mask_dir.resolve()
    data_root = data_root.resolve()
    if not output_mask_dir.is_dir():
        raise FileNotFoundError(f"MedSAM2 output-mask directory does not exist: {output_mask_dir}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"PolypGen sequence root does not exist: {data_root}")

    evaluation_dir = (evaluation_dir or output_mask_dir.parent / "evaluation").resolve()
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    sequence_names = sequences or discover_sequences(data_root)
    if not sequence_names:
        raise ValueError(f"No seqN directories found in {data_root}")

    sequence_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for sequence_name in sequence_names:
        sequence_row, rows = evaluate_sequence(data_root, output_mask_dir, evaluation_dir, sequence_name)
        sequence_rows.append(sequence_row)
        frame_rows.extend(rows)

    summary: dict[str, object] = {metric: mean_metric(frame_rows, metric) for metric in METRIC_NAMES}
    summary["sequences"] = len(sequence_rows)
    summary["frames"] = len(frame_rows)
    summary["missing_predictions"] = int(sum(row["missing_predictions"] for row in sequence_rows))

    write_csv(
        evaluation_dir / "metrics_per_sequence.csv",
        ["sequence", "frames", "missing_predictions", *METRIC_NAMES],
        sequence_rows,
    )
    write_csv(
        evaluation_dir / "metrics_per_frame.csv",
        ["sequence", "frame", "prediction_missing", *METRIC_NAMES],
        frame_rows,
    )
    stats_rows = []
    for aggregation, rows in (("frame", frame_rows), ("sequence", sequence_rows)):
        for metric in METRIC_NAMES:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            values = values[~np.isnan(values)]
            stats_rows.append({
                "aggregation": aggregation,
                "metric": metric,
                "mean": float(values.mean()) if values.size else float("nan"),
                "std": float(values.std()) if values.size else float("nan"),
                "min": float(values.min()) if values.size else float("nan"),
                "max": float(values.max()) if values.size else float("nan"),
            })
    write_csv(evaluation_dir / "metrics_stats.csv", ["aggregation", "metric", "mean", "std", "min", "max"], stats_rows)
    write_memory_confidence_statistics(output_mask_dir, evaluation_dir, sequence_names)
    for obsolete_path in (evaluation_dir / "metrics_avg.csv", evaluation_dir / "metrics_coverage.json"):
        obsolete_path.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MedSAM2 masks on PolypGen positive sequences.")
    parser.add_argument(
        "--output_mask_dir", "--pred_root", dest="output_mask_dir", required=True, type=Path,
        help="Directory passed as MedSAM2 --output_mask_dir (contains seqN mask folders).",
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT, help="PolypGen positive sequenceData root.")
    parser.add_argument(
        "--output_eval_dir", type=Path, default=None,
        help="Evaluation directory (default: sibling 'evaluation' next to output masks).",
    )
    parser.add_argument(
        "--sequences", nargs="*", default=None,
        help="Optional names, e.g. --sequences seq1 seq2. Defaults to every seqN directory.",
    )
    args = parser.parse_args()
    summary = evaluate_masks(args.output_mask_dir, args.data_root, args.output_eval_dir, args.sequences)
    print(
        "Evaluated {frames} frames from {sequences} sequences: "
        "Dice={dice:.4f}, IoU={iou:.4f}, F1={f_measure:.4f}, F2={f2:.4f}, "
        "missing predictions={missing_predictions}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
