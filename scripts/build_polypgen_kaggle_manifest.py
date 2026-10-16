#!/usr/bin/env python3
"""Build a manifest from the public PolypGen Video Sequence Kaggle layout."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adenoid.io import MASK_EXTENSIONS, resolve_mask_path


def numeric_order(paths):
    return sorted(paths, key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [args.data_root / "positive_cropped", args.data_root]
    sequence_root = next((root for root in roots if list(root.glob("seq*"))), None)
    if sequence_root is None:
        raise FileNotFoundError(f"No seq* directories under {args.data_root}")
    rows = []
    sequences = sorted(sequence_root.glob("seq*"), key=lambda path: int(re.search(r"\d+", path.name).group()))
    for sequence in sequences:
        image_dir = next((sequence / name for name in ("images", f"images_{sequence.name}") if (sequence / name).is_dir()), None)
        mask_dir = next((sequence / name for name in ("masks", "mask", "labels", "gt", f"masks_{sequence.name}") if (sequence / name).is_dir()), None)
        if image_dir is None or mask_dir is None:
            raise FileNotFoundError(f"Could not resolve image/mask directories for {sequence}")
        images = [path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        for index, image in enumerate(numeric_order(images)):
            mask = resolve_mask_path(mask_dir, image.stem)
            if mask is None:
                candidates = [mask_dir / f"{image.stem}_mask{extension}" for extension in MASK_EXTENSIONS]
                raise FileNotFoundError(f"No mask for {image}; tried {candidates}")
            rows.append({
                "split": "test", "video_id": sequence.name, "frame_index": index,
                "image": str(image.resolve()), "mask": str(mask.resolve()),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} frames from {len(sequences)} PolypGen sequences to {args.output}")


if __name__ == "__main__":
    main()
