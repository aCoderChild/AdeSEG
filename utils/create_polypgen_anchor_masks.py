"""Create one binary ground-truth anchor mask per PolypGen sequence."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image


def natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def image_paths(image_dir):
    return sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=natural_key,
    )


def create_anchor_masks(input_root, output_dir):
    rows = []
    sequence_dirs = sorted(
        (path for path in input_root.iterdir() if path.is_dir() and re.fullmatch(r"seq\d+", path.name)),
        key=natural_key,
    )
    for sequence_dir in sequence_dirs:
        sequence_name = sequence_dir.name
        image_dir = sequence_dir / f"images_{sequence_name}"
        mask_dir = sequence_dir / f"masks_{sequence_name}"
        anchor_dir = output_dir / sequence_name
        anchor_dir.mkdir(parents=True, exist_ok=True)

        anchor = None
        for frame_idx, image_path in enumerate(image_paths(image_dir)):
            mask_path = mask_dir / f"{image_path.stem}_mask.jpg"
            if not mask_path.exists():
                continue
            with Image.open(mask_path) as source_mask:
                binary_mask = np.asarray(source_mask.convert("L")) > 127
            if binary_mask.any():
                anchor = frame_idx, image_path, mask_path, binary_mask
                break

        if anchor is None:
            rows.append(
                {
                    "sequence": sequence_name,
                    "frame_idx": "",
                    "frame_name": "",
                    "source_mask": "",
                    "foreground_pixels": 0,
                    "status": "no_foreground_mask",
                }
            )
            continue

        frame_idx, image_path, mask_path, binary_mask = anchor
        anchor_path = anchor_dir / f"{image_path.stem}.png"
        Image.fromarray(binary_mask.astype(np.uint8)).save(anchor_path)
        rows.append(
            {
                "sequence": sequence_name,
                "frame_idx": frame_idx,
                "frame_name": image_path.stem,
                "source_mask": str(mask_path.relative_to(input_root)),
                "foreground_pixels": int(binary_mask.sum()),
                "status": "saved",
            }
        )

    manifest_path = output_dir / "anchors.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "sequence",
                "frame_idx",
                "frame_name",
                "source_mask",
                "foreground_pixels",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Save the first non-empty PolypGen ground-truth mask per sequence."
    )
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = create_anchor_masks(args.input_root, args.output_dir)
    saved_count = sum(row["status"] == "saved" for row in rows)
    print(f"Saved {saved_count}/{len(rows)} anchor masks to {args.output_dir}")


if __name__ == "__main__":
    main()
