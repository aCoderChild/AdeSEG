#!/usr/bin/env python3
"""Write a manifest for PolypGen protocol (a) positive sequences."""

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=ROOT / "data/PolypGen2021_MultiCenterData_v3/sequenceData/positive")
    parser.add_argument("--protocol", type=Path, default=ROOT / "datasets/splits/polypgen/protocol_a.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/polypgen_sequence.jsonl")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = []
    for split in ("dev", "test"):
        for number in protocol[split]:
            sequence = f"seq{number}"
            images = args.data_root / sequence / f"images_seq{number}"
            masks = args.data_root / sequence / f"masks_seq{number}"
            # Numeric frame order: an alphabetical sort puts frame 104 before 69.
            frames = sorted(images.glob("*.jpg"), key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)))
            for index, image in enumerate(frames):
                matches = sorted(path for path in masks.glob(f"{image.stem}_mask.*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
                if len(matches) != 1:
                    raise FileNotFoundError(f"Expected one mask for {image}")
                rows.append({
                    "split": split, "video_id": sequence, "frame_index": index,
                    "image": str(image.relative_to(output.parent)),
                    "mask": str(matches[0].relative_to(output.parent)),
                })
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} frames to {output}")


if __name__ == "__main__":
    main()
