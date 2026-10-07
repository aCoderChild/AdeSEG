#!/usr/bin/env python3
"""Convert an image/mask folder dataset into indexed masks and a JSONL manifest.

Layout: ``<root>/<split>/<image_dir>/*`` with masks of the same stem in
``<root>/<split>/<mask_dir>/``. Each image becomes a one-frame video. Mask grey
values are mapped to label ids with ``--value_map`` (to the nearest listed value,
so JPEG noise cannot create extra classes), for example:

  REFUGE (background, disc rim, cup):          --value_map 255:0 128:1 0:2
  Cai et al. 2024 adenoid (background, airway,
  adenoid):                                     --value_map 0:0 128:1 255:2
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_value_map(items: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pairs = [tuple(int(part) for part in item.split(":")) for item in items]
    values = np.array([value for value, _ in pairs], dtype=np.int16)
    labels = np.array([label for _, label in pairs], dtype=np.uint8)
    if len(set(values.tolist())) != len(values):
        raise ValueError("--value_map lists a grey value twice.")
    return values, labels


def to_labels(mask_path: Path, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    grey = np.asarray(Image.open(mask_path).convert("L"), dtype=np.int16)
    return labels[np.abs(grey[..., None] - values).argmin(axis=-1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--image_dir", default="images")
    parser.add_argument("--mask_dir", default="mask")
    parser.add_argument("--value_map", nargs="+", required=True, metavar="GREY:LABEL")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    values, labels = parse_value_map(args.value_map)
    manifest = args.output_dir / "manifest.jsonl"
    rows, counts = [], Counter()
    for split in args.splits:
        masks = {path.stem: path for path in (args.root / split / args.mask_dir).iterdir()
                 if path.suffix.lower() in IMAGE_SUFFIXES}
        out_dir = args.output_dir / "masks" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(path for path in (args.root / split / args.image_dir).iterdir()
                        if path.suffix.lower() in IMAGE_SUFFIXES)
        for image in images:
            if image.stem not in masks:
                raise FileNotFoundError(f"No mask for {image}")
            label_map = to_labels(masks[image.stem], values, labels)
            if label_map.shape != Image.open(image).size[::-1]:
                raise ValueError(f"Mask and image sizes differ for {image}")
            out_mask = out_dir / f"{image.stem}.png"
            Image.fromarray(label_map).save(out_mask)
            counts.update({int(label): 1 for label in np.unique(label_map)})
            rows.append({
                "split": split, "video_id": f"{split}_{image.stem}", "frame_index": 0,
                "image": os.path.relpath(image.resolve(), manifest.parent.resolve()),
                "mask": os.path.relpath(out_mask.resolve(), manifest.parent.resolve()),
            })
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} frames to {manifest}; frames containing each label: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
