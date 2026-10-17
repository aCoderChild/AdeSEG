#!/usr/bin/env python3
"""Build indexed CholecSeg8k clip manifests using the existing JSONL format.

The public release stores semantic labels as equal RGB grey values in
``*_endo_watershed_mask.png``.  This adapter validates that representation,
converts it once to indexed PNG masks, and retains the parent source-video ID
so later splits cannot leak clips across a source video.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image


# Published watershed greyscale codes, mapped to the 13 semantic class IDs.
WATERSHED_GREY_TO_CLASS = {
    80: 0,  # black background
    17: 1,  # abdominal wall
    33: 2,  # liver
    19: 3,  # gastrointestinal tract
    18: 4,  # fat
    49: 5,  # grasper
    35: 6,  # connective tissue
    36: 7,  # blood
    37: 8,  # cystic duct
    50: 9,  # L-hook electrocautery
    34: 10,  # gallbladder
    51: 11,  # hepatic vein
    5: 12,  # liver ligament
}


def frame_index(path: Path) -> int:
    match = re.fullmatch(r"frame_(\d+)_endo\.png", path.name)
    if match is None:
        raise ValueError(f"Unexpected CholecSeg8k image name: {path.name}")
    return int(match.group(1))


def indexed_watershed(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])):
        raise ValueError(f"Watershed mask is not grey-coded RGB: {path}")
    source = rgb[..., 0]
    values = set(np.unique(source).tolist())
    unknown = sorted(values - WATERSHED_GREY_TO_CLASS.keys())
    if unknown:
        raise ValueError(f"Unknown CholecSeg8k watershed codes {unknown} in {path}")
    indexed = np.zeros_like(source)
    for value, label in WATERSHED_GREY_TO_CLASS.items():
        indexed[source == value] = label
    return indexed


def clips(root: Path):
    for source_video in sorted(root.glob("video*")):
        if not source_video.is_dir():
            continue
        for clip in sorted(source_video.glob(f"{source_video.name}_*")):
            images = sorted(
                (
                    path
                    for path in clip.glob("frame_*_endo.png")
                    if "_mask" not in path.name and "_color_" not in path.name
                ),
                key=frame_index,
            )
            if not images:
                continue
            indices = [frame_index(path) for path in images]
            if indices != list(range(indices[0], indices[0] + len(indices))):
                raise ValueError(f"Non-consecutive frames in {clip}")
            yield source_video.name, clip.name, images


def build(root: Path, output_dir: Path, split: str = "analysis") -> list[dict]:
    root, output_dir = Path(root), Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest = output_dir / "cholecseg8k.jsonl"
    rows = []
    for source_video, clip_id, images in clips(root):
        for image in images:
            watershed = image.with_name(image.name.replace("_endo.png", "_endo_watershed_mask.png"))
            if not watershed.is_file():
                raise FileNotFoundError(watershed)
            labels = indexed_watershed(watershed)
            if labels.shape != Image.open(image).size[::-1]:
                raise ValueError(f"Image/mask size mismatch: {image}")
            indexed_path = output_dir / "masks" / clip_id / f"{image.stem}.png"
            indexed_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(labels).save(indexed_path)
            rows.append(
                {
                    "split": split,
                    "video_id": clip_id,
                    "source_video": source_video,
                    "frame_index": frame_index(image),
                    "image": os.path.relpath(image.resolve(), manifest.parent.resolve()),
                    "mask": os.path.relpath(indexed_path.resolve(), manifest.parent.resolve()),
                }
            )
    if not rows:
        raise ValueError(f"No CholecSeg8k clips found under {root}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="analysis")
    args = parser.parse_args()
    rows = build(args.data_root, args.output_dir, args.split)
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "cholecseg8k.jsonl"),
                "frames": len(rows),
                "clips": len({row["video_id"] for row in rows}),
                "source_videos": sorted({row["source_video"] for row in rows}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
