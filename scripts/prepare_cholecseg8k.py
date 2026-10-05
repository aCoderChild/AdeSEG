#!/usr/bin/env python3
"""Convert CholecSeg8k into the PolypGen sequence layout for single-object video segmentation.

CholecSeg8k (CC BY-NC-SA 4.0): 101 clips of 80 consecutive laparoscopic frames from
17 Cholec80 videos, each frame with a watershed mask of 13 classes. For one target
class, every clip containing it becomes ``<output>/seqN/images_seqN/<frame>.jpg``
(MedSAM2's video loader reads JPEG) and ``masks_seqN/<frame>_mask.png`` (binary), so
``scripts/infer.py -i <output>`` and ``evaluation/temporal.py --data_root <output>``
run unchanged. ``sequences.csv`` maps seqN to the source clip, and ``split.json``
splits whole surgical videos (not clips) into dev and test.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASS_VALUES = {  # watershed-mask grey values, from the CholecSeg8k paper
    "abdominal_wall": 11, "fat": 12, "gastrointestinal_tract": 13, "liver": 21, "gallbladder": 22,
    "connective_tissue": 23, "blood": 24, "cystic_duct": 25, "grasper": 31, "l_hook_electrocautery": 32,
    "hepatic_vein": 33, "liver_ligament": 5,
}


def frame_number(path: Path) -> int:
    return int(re.search(r"frame_(\d+)_endo", path.name).group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/CholecSeg8k")
    parser.add_argument("--output", type=Path, default=ROOT / "data/CholecSeg8k_gallbladder")
    parser.add_argument("--target", default="gallbladder", choices=sorted(CLASS_VALUES))
    parser.add_argument("--min_pixels", type=int, default=200, help="Smaller target regions count as absent.")
    parser.add_argument("--test_fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    value = CLASS_VALUES[args.target]

    clips = sorted(p for p in args.source.glob("video*/video*_*") if p.is_dir())
    rows, number = [], 0
    for clip in clips:
        images = sorted(clip.glob("frame_*_endo.png"), key=frame_number)
        masks = []
        for image in images:
            watershed = np.array(Image.open(image.with_name(image.stem + "_watershed_mask.png")).convert("L"))
            target = watershed == value
            masks.append(target if target.sum() >= args.min_pixels else np.zeros_like(target))
        present = [bool(m.any()) for m in masks]
        if not any(present):
            continue
        number += 1
        name = f"seq{number}"
        image_dir, mask_dir = args.output / name / f"images_{name}", args.output / name / f"masks_{name}"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for image, mask in zip(images, masks):
            frame = f"{clip.name}_{frame_number(image):05d}"
            Image.open(image).convert("RGB").save(image_dir / f"{frame}.jpg", quality=95)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / f"{frame}_mask.png")
        rows.append({"sequence": name, "video": clip.parent.name, "clip": clip.name, "frames": len(images),
                     "present_frames": sum(present)})
        print(name, clip.name, f"{sum(present)}/{len(images)} frames with {args.target}", flush=True)

    with open(args.output / "sequences.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    videos = sorted({r["video"] for r in rows})
    rng = np.random.default_rng(args.seed)
    test_videos = set(rng.choice(videos, size=max(1, round(len(videos) * args.test_fraction)), replace=False))
    split = {
        "description": f"CholecSeg8k {args.target}: whole surgical videos split into dev and test (seed {args.seed}).",
        "test_videos": sorted(test_videos),
        "dev": [int(r["sequence"][3:]) for r in rows if r["video"] not in test_videos],
        "test": [int(r["sequence"][3:]) for r in rows if r["video"] in test_videos],
    }
    (args.output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    print(f"{len(rows)} sequences from {len(videos)} videos; dev {len(split['dev'])}, test {len(split['test'])}")


if __name__ == "__main__":
    main()
