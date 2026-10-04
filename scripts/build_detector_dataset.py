#!/usr/bin/env python3
"""Build a YOLO polyp-detection dataset from the PolypGen training data only.

Positives: ``data_C1``..``data_C6`` single frames minus
``datasets/splits/polypgen/single_frame_test_overlap.txt``; one box per connected
mask region. Background: a subsample of ``sequenceData/negativeOnly`` frames with
empty label files. No ``sequenceData/positive`` frame is used. Images are
symlinked; labels are YOLO ``class cx cy w h`` (normalized).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.pseudo_video import list_negative_sequences, list_single_frames, load_excluded_paths
from adenoid.io import load_binary_mask


def boxes_from_mask(mask: np.ndarray, min_area_fraction: float) -> list[tuple[float, float, float, float]]:
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:count]:
        if area >= min_area_fraction * height * width:
            boxes.append(((x + w / 2) / width, (y + h / 2) / height, w / width, h / height))
    return boxes


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(source.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polypgen_root", type=Path, default=ROOT / "data/PolypGen2021_MultiCenterData_v3")
    parser.add_argument(
        "--exclude_list", type=Path, default=ROOT / "datasets/splits/polypgen/single_frame_test_overlap.txt"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative_stride", type=int, default=3, help="Use every n-th negativeOnly frame.")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--min_area_fraction", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    positives = list_single_frames(args.polypgen_root, excluded=load_excluded_paths(args.exclude_list))
    negatives = [path for sequence in list_negative_sequences(args.polypgen_root) for path in sequence]
    negatives = negatives[:: args.negative_stride]
    counts = {"train": [0, 0, 0], "val": [0, 0, 0]}  # positive images, boxes, background images

    for index, frame in enumerate(positives):
        split = "val" if rng.random() < args.val_fraction else "train"
        name = f"{frame.center}_{frame.image_path.stem}"
        boxes = boxes_from_mask(load_binary_mask(frame.mask_path), args.min_area_fraction)
        link(frame.image_path, args.output / "images" / split / f"{name}{frame.image_path.suffix}")
        label = args.output / "labels" / split / f"{name}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes))
        counts[split][0] += 1
        counts[split][1] += len(boxes)

    for path in negatives:
        split = "val" if rng.random() < args.val_fraction else "train"
        name = f"neg_{path.parent.name}_{path.stem}"
        link(path, args.output / "images" / split / f"{name}{path.suffix}")
        label = args.output / "labels" / split / f"{name}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("")
        counts[split][2] += 1

    (args.output / "polyp.yaml").write_text(
        f"path: {args.output.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: polyp\n"
    )
    summary = {
        split: {"positive_images": c[0], "boxes": c[1], "background_images": c[2]} for split, c in counts.items()
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
