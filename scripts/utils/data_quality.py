#!/usr/bin/env python3
"""Score raw PolypGen video quality before model inference.

The audit writes numeric quality scores, not categorical labels. It uses
external/IQA-PyTorch for BRISQUE, NIQE, MUSIQ, or other pyiqa metrics when that
package is importable. It can also summarize Endoscopic-Artefact-Detection text
predictions when they are provided. Lightweight local scores are implemented
only for the simple frame/video quantities that are not exposed by those
external folders as reusable functions.
"""

import argparse
import csv
import glob
import types
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
EPSILON = 1e-12
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PYIQA_METRICS = ("brisque", "niqe", "musiq")
EAD_CLASSES = (
    "specularity",
    "saturation",
    "artifact",
    "blur",
    "contrast",
    "bubbles",
    "instrument",
    "blood",
)


def numeric_sort_key(path: str) -> Tuple[object, ...]:
    stem = os.path.splitext(os.path.basename(path))[0]
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", stem))


def list_sequences(data_root: str) -> List[str]:
    seq_dirs = [
        path
        for path in glob.glob(os.path.join(data_root, "seq*"))
        if os.path.isdir(path)
    ]
    return [os.path.basename(path) for path in sorted(seq_dirs, key=numeric_sort_key)]


def list_images(directory: str) -> List[str]:
    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    return sorted(paths, key=numeric_sort_key)


def resolve_mask_path(mask_dir: str, frame_stem: str) -> Optional[str]:
    for ext in MASK_EXTENSIONS:
        path = os.path.join(mask_dir, f"{frame_stem}{ext}")
        if os.path.exists(path):
            return path
    return None


def read_color(path: str) -> Optional[np.ndarray]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def read_mask(mask_dir: str, frame_stem: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    path = resolve_mask_path(mask_dir, frame_stem)
    if path is None:
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def entropy_from_values(values: np.ndarray, bins: int = 256) -> float:
    hist, _ = np.histogram(values.ravel(), bins=bins, range=(0, 255), density=False)
    probabilities = hist.astype(np.float64)
    probabilities /= np.sum(probabilities) + EPSILON
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad_sharpness(gray: np.ndarray) -> float:
    gray_float = gray.astype(np.float32)
    sobel_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(sobel_x * sobel_x + sobel_y * sobel_y))


def rms_contrast(gray: np.ndarray) -> float:
    return float(np.std(gray.astype(np.float32)))


def normalized_mean(values: np.ndarray) -> float:
    return float(np.mean(values.astype(np.float32)) / 255.0)


def artifact_score_masks(image_bgr: np.ndarray, gray: np.ndarray) -> Dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    max_channel = np.max(image_bgr, axis=2)
    min_channel = np.min(image_bgr, axis=2)

    # Conservative handcrafted proxies. If EAD predictions are supplied, those
    # are reported separately rather than fused with these local scores.
    specular = (value >= 235) & (saturation <= 70) & ((max_channel - min_channel) <= 55)
    saturated = (max_channel >= 250) | (saturation >= 245)
    dark = gray <= 35
    return {
        "specular": specular,
        "saturated": saturated,
        "dark": dark,
    }


def binary_fraction(mask: np.ndarray, region: Optional[np.ndarray] = None) -> float:
    if region is None:
        return float(np.mean(mask))
    if not np.any(region):
        return float("nan")
    return float(np.mean(mask[region]))


def score_region(gray: np.ndarray, region: Optional[np.ndarray]) -> Dict[str, float]:
    if region is None or not np.any(region):
        return {
            "region_laplacian_var": float("nan"),
            "region_contrast": float("nan"),
            "region_entropy": float("nan"),
        }
    values = gray[region]
    region_gray = np.zeros_like(gray)
    region_gray[region] = gray[region]
    return {
        "region_laplacian_var": laplacian_variance(region_gray),
        "region_contrast": float(np.std(values.astype(np.float32))),
        "region_entropy": entropy_from_values(values),
    }


def frame_motion(prev_gray: Optional[np.ndarray], gray: np.ndarray) -> float:
    if prev_gray is None:
        return float("nan")
    if prev_gray.shape != gray.shape:
        prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))
    diff = cv2.absdiff(prev_gray, gray)
    return float(np.mean(diff))


def simple_frame_scores(
    image_bgr: np.ndarray,
    gray: np.ndarray,
    mask: Optional[np.ndarray],
    prev_gray: Optional[np.ndarray],
) -> Dict[str, float]:
    masks = artifact_score_masks(image_bgr, gray)
    scores = {
        "laplacian_blur_score": laplacian_variance(gray),
        "tenengrad_sharpness": tenengrad_sharpness(gray),
        "dark_region_ratio": binary_fraction(masks["dark"]),
        "saturation_ratio": binary_fraction(masks["saturated"]),
        "specular_ratio": binary_fraction(masks["specular"]),
        "contrast_rms": rms_contrast(gray),
        "entropy": entropy_from_values(gray),
        "brightness_mean": normalized_mean(gray),
        "motion_frame_diff": frame_motion(prev_gray, gray),
    }
    if mask is not None:
        scores["mask_area_fraction"] = float(np.mean(mask))
        scores["mask_specular_ratio"] = binary_fraction(masks["specular"], mask)
        scores["mask_saturation_ratio"] = binary_fraction(masks["saturated"], mask)
        scores["mask_dark_ratio"] = binary_fraction(masks["dark"], mask)
        region_scores = score_region(gray, mask)
        scores["mask_laplacian_blur_score"] = region_scores["region_laplacian_var"]
        scores["mask_contrast_rms"] = region_scores["region_contrast"]
        scores["mask_entropy"] = region_scores["region_entropy"]
    else:
        scores["mask_area_fraction"] = float("nan")
        scores["mask_specular_ratio"] = float("nan")
        scores["mask_saturation_ratio"] = float("nan")
        scores["mask_dark_ratio"] = float("nan")
        scores["mask_laplacian_blur_score"] = float("nan")
        scores["mask_contrast_rms"] = float("nan")
        scores["mask_entropy"] = float("nan")
    return scores


@dataclass
class PyiqaRunner:
    root: str
    metric_names: Sequence[str]
    device: Optional[str]
    enabled: bool = True

    def __post_init__(self) -> None:
        self.models = {}
        if not self.enabled or not self.metric_names:
            return
        if not os.path.isdir(self.root):
            print(f"Warning: IQA-PyTorch root not found: {self.root}", file=sys.stderr)
            return
        torch_cache = os.path.join(PROJECT_ROOT, ".cache", "torch")
        os.environ.setdefault("TORCH_HOME", torch_cache)
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, ".cache"))
        os.makedirs(torch_cache, exist_ok=True)
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        try:
            import pyiqa  # type: ignore
        except Exception as exc:
            if "pyiqa.version" not in str(exc):
                print(f"Warning: could not import pyiqa from {self.root}: {exc}", file=sys.stderr)
                return
            version_module = types.ModuleType("pyiqa.version")
            version_module.__version__ = "local"
            sys.modules["pyiqa.version"] = version_module
            try:
                import pyiqa  # type: ignore
            except Exception as retry_exc:
                print(
                    f"Warning: could not import pyiqa from {self.root}: {retry_exc}",
                    file=sys.stderr,
                )
                return
        for metric_name in self.metric_names:
            try:
                self.models[metric_name] = pyiqa.create_metric(metric_name, device=self.device)
            except Exception as exc:
                print(
                    f"Warning: could not initialize pyiqa metric '{metric_name}': {exc}",
                    file=sys.stderr,
                )

    @property
    def fields(self) -> List[str]:
        if not self.enabled:
            return []
        return [f"pyiqa_{name}" for name in self.metric_names]

    def score(self, image_path: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for metric_name in self.metric_names:
            model = self.models.get(metric_name)
            if model is None:
                scores[f"pyiqa_{metric_name}"] = float("nan")
                continue
            try:
                value = model(image_path).detach().cpu().item()
            except Exception as exc:
                print(
                    f"Warning: pyiqa metric '{metric_name}' failed on {image_path}: {exc}",
                    file=sys.stderr,
                )
                value = float("nan")
            scores[f"pyiqa_{metric_name}"] = float(value)
        return scores


class EadPredictionReader:
    """Read Endoscopic-Artefact-Detection prediction txt files if supplied.

    Expected rows follow the EAD repository convention:
    class_name confidence x1 y1 x2 y2
    """

    def __init__(self, prediction_root: Optional[str]) -> None:
        self.prediction_root = prediction_root

    @property
    def enabled(self) -> bool:
        return bool(self.prediction_root and os.path.isdir(self.prediction_root))

    @property
    def fields(self) -> List[str]:
        fields: List[str] = []
        for class_name in EAD_CLASSES:
            fields.extend(
                [
                    f"ead_{class_name}_count",
                    f"ead_{class_name}_max_conf",
                    f"ead_{class_name}_area_ratio",
                ]
            )
        return fields

    def _candidate_paths(self, seq: str, frame_stem: str) -> List[str]:
        assert self.prediction_root is not None
        return [
            os.path.join(self.prediction_root, seq, f"{frame_stem}.txt"),
            os.path.join(self.prediction_root, f"{seq}_{frame_stem}.txt"),
            os.path.join(self.prediction_root, f"{frame_stem}.txt"),
        ]

    def score(self, seq: str, frame_stem: str, image_shape: Tuple[int, int]) -> Dict[str, float]:
        scores = {field: float("nan") for field in self.fields}
        if not self.enabled:
            return scores

        prediction_path = next(
            (path for path in self._candidate_paths(seq, frame_stem) if os.path.exists(path)),
            None,
        )
        if prediction_path is None:
            return scores

        height, width = image_shape
        image_area = float(height * width)
        by_class: Dict[str, List[Tuple[float, float]]] = {name: [] for name in EAD_CLASSES}
        with open(prediction_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                class_name = parts[0].lower()
                if class_name not in by_class:
                    continue
                try:
                    confidence = float(parts[1])
                    x1, y1, x2, y2 = [float(value) for value in parts[2:6]]
                except ValueError:
                    continue
                x1 = min(max(x1, 0.0), width)
                x2 = min(max(x2, 0.0), width)
                y1 = min(max(y1, 0.0), height)
                y2 = min(max(y2, 0.0), height)
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (image_area + EPSILON)
                by_class[class_name].append((confidence, area))

        for class_name, detections in by_class.items():
            if not detections:
                scores[f"ead_{class_name}_count"] = 0.0
                scores[f"ead_{class_name}_max_conf"] = 0.0
                scores[f"ead_{class_name}_area_ratio"] = 0.0
                continue
            scores[f"ead_{class_name}_count"] = float(len(detections))
            scores[f"ead_{class_name}_max_conf"] = float(max(conf for conf, _ in detections))
            scores[f"ead_{class_name}_area_ratio"] = float(sum(area for _, area in detections))
        return scores


def mean_or_empty(values: Iterable[Optional[float]]) -> str:
    numeric = [float(value) for value in values if value is not None and not np.isnan(value)]
    if not numeric:
        return ""
    return f"{np.mean(numeric):.6f}"


def min_or_empty(values: Iterable[Optional[float]]) -> str:
    numeric = [float(value) for value in values if value is not None and not np.isnan(value)]
    if not numeric:
        return ""
    return f"{np.min(numeric):.6f}"


def max_or_empty(values: Iterable[Optional[float]]) -> str:
    numeric = [float(value) for value in values if value is not None and not np.isnan(value)]
    if not numeric:
        return ""
    return f"{np.max(numeric):.6f}"


def std_or_empty(values: Iterable[Optional[float]]) -> str:
    numeric = [float(value) for value in values if value is not None and not np.isnan(value)]
    if not numeric:
        return ""
    return f"{np.std(numeric):.6f}"


def format_float(value: object) -> object:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6f}"
    return value


def audit_sequence(
    data_root: str,
    seq: str,
    pyiqa_runner: PyiqaRunner,
    ead_reader: EadPredictionReader,
    pyiqa_stride: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    images_dir = os.path.join(data_root, seq, "images")
    masks_dir = os.path.join(data_root, seq, "masks")
    image_paths = list_images(images_dir)
    prev_gray: Optional[np.ndarray] = None
    frame_rows: List[Dict[str, object]] = []

    for frame_idx, image_path in enumerate(image_paths):
        image = read_color(image_path)
        if image is None:
            continue
        frame_name = os.path.basename(image_path)
        frame_stem = os.path.splitext(frame_name)[0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = read_mask(masks_dir, frame_stem, gray.shape) if os.path.isdir(masks_dir) else None

        row: Dict[str, object] = {
            "sequence": seq,
            "frame": frame_name,
            "frame_idx": frame_idx,
            "height": gray.shape[0],
            "width": gray.shape[1],
        }
        row.update(simple_frame_scores(image, gray, mask, prev_gray))

        if pyiqa_stride > 0 and frame_idx % pyiqa_stride == 0:
            row.update(pyiqa_runner.score(image_path))
        else:
            row.update({field: float("nan") for field in pyiqa_runner.fields})

        if ead_reader.enabled:
            row.update(ead_reader.score(seq, frame_stem, gray.shape))
        frame_rows.append(row)
        prev_gray = gray

    numeric_fields = [
        field
        for row in frame_rows
        for field, value in row.items()
        if field not in {"sequence", "frame", "frame_idx", "height", "width"}
        and isinstance(value, (int, float, np.integer, np.floating))
    ]
    numeric_fields = sorted(set(numeric_fields))
    sequence_row: Dict[str, object] = {
        "sequence": seq,
        "frame_count": len(frame_rows),
    }
    if frame_rows:
        sequence_row["height"] = frame_rows[0].get("height", "")
        sequence_row["width"] = frame_rows[0].get("width", "")
    for field in numeric_fields:
        values = [row.get(field) for row in frame_rows]
        sequence_row[f"mean_{field}"] = mean_or_empty(values)
        sequence_row[f"min_{field}"] = min_or_empty(values)
        sequence_row[f"max_{field}"] = max_or_empty(values)
        sequence_row[f"std_{field}"] = std_or_empty(values)
    return sequence_row, frame_rows


def collect_fieldnames(rows: Sequence[Dict[str, object]], preferred: Sequence[str]) -> List[str]:
    seen = set()
    fields: List[str] = []
    for field in preferred:
        if any(field in row for row in rows):
            fields.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_float(row.get(field, "")) for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/polypgen", help="Root containing seq*/images and seq*/masks")
    parser.add_argument("--output_csv", default="outputs/data_quality_report.csv")
    parser.add_argument("--frame_output_csv", default="outputs/data_quality_frame_scores.csv")
    parser.add_argument("--seqs", nargs="*", help="Optional sequence names or numbers, e.g. 1 2 seq3")
    parser.add_argument("--iqa_pytorch_root", default="external/IQA-PyTorch")
    parser.add_argument("--pyiqa_metrics", nargs="*", default=list(DEFAULT_PYIQA_METRICS))
    parser.add_argument("--pyiqa_device", default=None, help="Device passed to pyiqa, e.g. cpu, cuda, mps")
    parser.add_argument(
        "--pyiqa_stride",
        type=int,
        default=1,
        help="Run pyiqa every N frames. Use 0 to skip pyiqa scoring.",
    )
    parser.add_argument(
        "--ead_predictions_dir",
        default=None,
        help="Optional Endoscopic-Artefact-Detection txt prediction root.",
    )
    return parser.parse_args()


def normalize_sequence(seq: str) -> str:
    return seq if seq.startswith("seq") else f"seq{seq}"


def main() -> None:
    args = parse_args()
    sequences = [normalize_sequence(seq) for seq in args.seqs] if args.seqs else list_sequences(args.data_root)

    pyiqa_enabled = args.pyiqa_stride > 0
    pyiqa_runner = PyiqaRunner(
        root=args.iqa_pytorch_root,
        metric_names=args.pyiqa_metrics,
        device=args.pyiqa_device,
        enabled=pyiqa_enabled,
    )
    ead_reader = EadPredictionReader(args.ead_predictions_dir)
    if args.ead_predictions_dir and not ead_reader.enabled:
        print(f"Warning: EAD prediction directory not found: {args.ead_predictions_dir}", file=sys.stderr)

    sequence_rows: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    for seq in sequences:
        sequence_row, seq_frame_rows = audit_sequence(
            data_root=args.data_root,
            seq=seq,
            pyiqa_runner=pyiqa_runner,
            ead_reader=ead_reader,
            pyiqa_stride=args.pyiqa_stride,
        )
        sequence_rows.append(sequence_row)
        frame_rows.extend(seq_frame_rows)

    sequence_rows.sort(key=lambda row: numeric_sort_key(str(row["sequence"])))
    frame_rows.sort(key=lambda row: (numeric_sort_key(str(row["sequence"])), int(row["frame_idx"])))

    sequence_fields = collect_fieldnames(sequence_rows, preferred=["sequence", "frame_count"])
    frame_fields = collect_fieldnames(
        frame_rows,
        preferred=["sequence", "frame", "frame_idx", "height", "width"],
    )

    write_csv(args.output_csv, sequence_rows, sequence_fields)
    write_csv(args.frame_output_csv, frame_rows, frame_fields)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.frame_output_csv}")


if __name__ == "__main__":
    main()
