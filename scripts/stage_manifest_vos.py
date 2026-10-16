#!/usr/bin/env python3
"""Stage a manifest split as DAVIS-style JPEGImages/Annotations symlinks."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    videos = load_manifest(args.manifest, args.split)
    rows = []
    for video, frames in videos.items():
        images = args.output_dir / "JPEGImages" / video
        masks = args.output_dir / "Annotations" / video
        images.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            name = f"{index:05d}"
            (images / f"{name}{frame.image_path.suffix.lower()}").symlink_to(frame.image_path.resolve())
            (masks / f"{name}.png").symlink_to(frame.mask_path.resolve())
            rows.append({
                "split": args.split, "video_id": video, "frame_index": index,
                "image": str((images / f"{name}{frame.image_path.suffix.lower()}").absolute()),
                "mask": str((masks / f"{name}.png").absolute()),
            })
    (args.output_dir / "videos.txt").write_text("\n".join(videos) + "\n")
    (args.output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    print(f"Staged {len(videos)} videos under {args.output_dir}")


if __name__ == "__main__":
    main()
