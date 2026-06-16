#!/usr/bin/env python3
"""Compute ground-truth mask IoU over a bounded temporal lookback window."""

import argparse
import csv
import glob
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


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


def mask_path_for_frame(mask_dir: str, image_path: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for ext in IMAGE_EXTENSIONS:
        path = os.path.join(mask_dir, f"{stem}{ext}")
        if os.path.exists(path):
            return path
    return None


def read_binary_mask(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 0


def resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return cv2.resize(
        mask.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0


def temporal_iou(current_mask: np.ndarray, previous_mask: np.ndarray) -> float:
    previous_mask = resize_mask(previous_mask, current_mask.shape)
    union = np.logical_or(current_mask, previous_mask).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(current_mask, previous_mask).sum()
    return float(intersection / union)


def multiframe_iou(masks: Sequence[np.ndarray]) -> float:
    if not masks:
        return 1.0
    base_shape = masks[-1].shape
    aligned_masks = [resize_mask(mask, base_shape) for mask in masks]
    intersection = np.logical_and.reduce(aligned_masks).sum()
    union = np.logical_or.reduce(aligned_masks).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def mean_or_empty(values: Iterable[float]) -> str:
    values = list(values)
    if not values:
        return ""
    return str(float(np.mean(values)))


def std_or_empty(values: Iterable[float]) -> str:
    values = list(values)
    if not values:
        return ""
    return str(float(np.std(values)))


def median_or_empty(values: Iterable[float]) -> str:
    values = list(values)
    if not values:
        return ""
    return str(float(np.median(values)))


def min_or_empty(values: Iterable[float]) -> str:
    values = list(values)
    if not values:
        return ""
    return str(float(np.min(values)))


def max_or_empty(values: Iterable[float]) -> str:
    values = list(values)
    if not values:
        return ""
    return str(float(np.max(values)))


def normalize_sequence(seq: str) -> str:
    return seq if seq.startswith("seq") else f"seq{seq}"


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_values(values: List[float]) -> Dict[str, object]:
    return {
        "num_windows": len(values),
        "mean_gt_temporal_iou": mean_or_empty(values),
        "std_gt_temporal_iou": std_or_empty(values),
        "median_gt_temporal_iou": median_or_empty(values),
        "min_gt_temporal_iou": min_or_empty(values),
        "max_gt_temporal_iou": max_or_empty(values),
    }


def load_sequence_masks(data_root: str, seq: str) -> Tuple[List[str], List[np.ndarray]]:
    images_dir = os.path.join(data_root, seq, "images")
    masks_dir = os.path.join(data_root, seq, "masks")
    image_paths = list_images(images_dir)

    masks: List[np.ndarray] = []
    frame_names: List[str] = []
    for image_path in image_paths:
        mask = read_binary_mask(mask_path_for_frame(masks_dir, image_path))
        if mask is None:
            continue
        masks.append(mask)
        frame_names.append(os.path.splitext(os.path.basename(image_path))[0])
    return frame_names, masks


def compute_pairwise_sequence(data_root: str, seq: str, max_lookback: int) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    frame_names, masks = load_sequence_masks(data_root, seq)

    rows: List[Dict[str, object]] = []
    values: List[float] = []
    pair_index = 1
    for current_idx, current_mask in enumerate(masks):
        previous_start = max(0, current_idx - max_lookback)
        for previous_idx in range(previous_start, current_idx):
            iou = temporal_iou(current_mask, masks[previous_idx])
            values.append(iou)
            rows.append(
                {
                    "sequence": seq,
                    "pair_index": pair_index,
                    "previous_frame": frame_names[previous_idx],
                    "current_frame": frame_names[current_idx],
                    "previous_index": previous_idx,
                    "current_index": current_idx,
                    "lookback": current_idx - previous_idx,
                    "gt_temporal_iou": iou,
                }
            )
            pair_index += 1

    summary = {
        "sequence": seq,
        "num_masks": len(masks),
        "max_lookback": max_lookback,
        **summarize_values(values),
    }
    return summary, rows


def compute_multiframe_sequence(data_root: str, seq: str, max_lookback: int) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    frame_names, masks = load_sequence_masks(data_root, seq)

    rows: List[Dict[str, object]] = []
    values: List[float] = []
    for current_idx in range(1, len(masks)):
        window_start = max(0, current_idx - max_lookback)
        window_masks = masks[window_start : current_idx + 1]
        iou = multiframe_iou(window_masks)
        values.append(iou)
        rows.append(
            {
                "sequence": seq,
                "window_index": len(rows) + 1,
                "window_start_frame": frame_names[window_start],
                "window_end_frame": frame_names[current_idx],
                "window_start_index": window_start,
                "window_end_index": current_idx,
                "num_frames": len(window_masks),
                "num_previous_frames": current_idx - window_start,
                "gt_temporal_iou": iou,
            }
        )

    summary = {
        "sequence": seq,
        "num_masks": len(masks),
        "max_lookback": max_lookback,
        **summarize_values(values),
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/polypgen", help="Root containing seq*/images and seq*/masks")
    parser.add_argument("--output_dir", default="outputs/ground_truth")
    parser.add_argument("--max_lookback", type=int, default=7, help="Maximum number of previous frames per current frame")
    parser.add_argument("--seqs", nargs="*", help="Optional sequence names or numbers, e.g. 1 2 seq3")
    parser.add_argument("--prefix", default="gt_temporal_iou_window7")
    parser.add_argument(
        "--mode",
        choices=["pairwise", "multiframe"],
        default="pairwise",
        help="pairwise compares each current frame to each previous frame; multiframe computes one IoU over the whole window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = [normalize_sequence(seq) for seq in args.seqs] if args.seqs else list_sequences(args.data_root)

    per_sequence_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    compute_sequence = compute_pairwise_sequence if args.mode == "pairwise" else compute_multiframe_sequence
    for seq in sequences:
        summary, rows = compute_sequence(args.data_root, seq, args.max_lookback)
        per_sequence_rows.append(summary)
        detail_rows.extend(rows)

    detail_values = [float(row["gt_temporal_iou"]) for row in detail_rows]
    global_summary = {
        "num_sequences": len(per_sequence_rows),
        "max_lookback": args.max_lookback,
        **summarize_values(detail_values),
    }

    pair_fields = [
        "sequence",
        "pair_index",
        "previous_frame",
        "current_frame",
        "previous_index",
        "current_index",
        "lookback",
        "gt_temporal_iou",
    ]
    per_sequence_fields = [
        "sequence",
        "num_masks",
        "max_lookback",
        "num_windows",
        "mean_gt_temporal_iou",
        "std_gt_temporal_iou",
        "median_gt_temporal_iou",
        "min_gt_temporal_iou",
        "max_gt_temporal_iou",
    ]
    summary_fields = [
        "num_sequences",
        "max_lookback",
        "num_windows",
        "mean_gt_temporal_iou",
        "std_gt_temporal_iou",
        "median_gt_temporal_iou",
        "min_gt_temporal_iou",
        "max_gt_temporal_iou",
    ]
    window_fields = [
        "sequence",
        "window_index",
        "window_start_frame",
        "window_end_frame",
        "window_start_index",
        "window_end_index",
        "num_frames",
        "num_previous_frames",
        "gt_temporal_iou",
    ]

    detail_suffix = "pairs" if args.mode == "pairwise" else "windows"
    detail_fields = pair_fields if args.mode == "pairwise" else window_fields
    detail_path = os.path.join(args.output_dir, f"{args.prefix}_{detail_suffix}.csv")
    per_sequence_path = os.path.join(args.output_dir, f"{args.prefix}_per_sequence.csv")
    summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.csv")

    write_csv(detail_path, detail_rows, detail_fields)
    write_csv(per_sequence_path, per_sequence_rows, per_sequence_fields)
    write_csv(summary_path, [global_summary], summary_fields)

    print(f"Wrote {detail_path}")
    print(f"Wrote {per_sequence_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
