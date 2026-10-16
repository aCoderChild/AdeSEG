#!/usr/bin/env python3
"""Convert the official DAVIS 2017 train/val layout to AdeSEG's video manifest."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing DAVIS/JPEGImages/480p and DAVIS/Annotations/480p.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    davis = args.data_root / "DAVIS" if (args.data_root / "DAVIS").is_dir() else args.data_root
    protocol = json.loads(args.protocol.read_text())["davis_2017"]
    holdout = set(protocol["model_selection_holdout"])
    groups = {
        "train": [name for name in protocol["train_sequences"] if name not in holdout],
        "holdout": protocol["model_selection_holdout"],
        "val": protocol["validation_sequences"],
    }
    rows = []
    output = args.output.resolve()
    for split, sequences in groups.items():
        for sequence in sequences:
            images = sorted((davis / "JPEGImages" / "480p" / sequence).glob("*.jpg"))
            if not images:
                raise FileNotFoundError(f"No DAVIS frames for {sequence}")
            for index, image in enumerate(images):
                mask = davis / "Annotations" / "480p" / sequence / f"{image.stem}.png"
                if not mask.is_file():
                    raise FileNotFoundError(mask)
                rows.append({
                    "split": split, "video_id": sequence, "frame_index": index,
                    "image": str(image.resolve()), "mask": str(mask.resolve()),
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} frames across {sum(map(len, groups.values()))} DAVIS sequences to {output}")


if __name__ == "__main__":
    main()
