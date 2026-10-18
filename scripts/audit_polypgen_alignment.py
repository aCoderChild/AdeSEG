#!/usr/bin/env python3
"""Compare downloaded mirror samples with original local PolypGen frames.

This is provenance analysis, not a model or a learned correspondence method.
Report raw numerical nearest-image matches; do not claim byte equivalence from
approximate pixel agreement. Remote samples must match the run's saved hashes.
"""

# ruff: noqa: E402
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from scripts.run_region_experiments import validate_output
from scripts.run_region_token_pilot import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl"
    )
    parser.add_argument("--video", default="seq13")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = validate_output(args.output)
    frames = load_manifest(args.manifest, "dev")[args.video]
    inventory = {
        row["frame_index"]: row
        for row in json.loads(args.inventory.read_text())["files"]
        if row["video_id"] == args.video
    }
    thumbnails = [
        np.asarray(
            Image.open(f.image_path).convert("RGB").resize((32, 32)), dtype=np.float32
        )
        for f in frames
    ]
    matches = []
    for sample in sorted(args.sample_root.glob(f"remote_{args.video}_frame*/*.jpg")):
        remote_index = int(sample.stem)
        observed_hash = hashlib.sha256(sample.read_bytes()).hexdigest()
        if observed_hash != inventory[remote_index]["image_sha256"]:
            raise ValueError(
                "Downloaded sample differs from running notebook's saved inventory"
            )
        image = Image.open(sample).convert("RGB")
        thumbnail = np.asarray(image.resize((32, 32)), dtype=np.float32)
        scores = [
            float(np.mean((thumbnail - reference) ** 2)) for reference in thumbnails
        ]
        ranking = np.argsort(scores)
        index = int(ranking[0])
        reference = Image.open(frames[index].image_path).convert("RGB")
        mae = (
            float(
                np.abs(
                    np.asarray(image, dtype=np.float32)
                    - np.asarray(reference, dtype=np.float32)
                ).mean()
            )
            if image.size == reference.size
            else None
        )
        matches.append(
            {
                "remote_index": remote_index,
                "remote_sha256": observed_hash,
                "remote_matches_saved_inventory": True,
                "nearest_local_index": index,
                "nearest_original_file": frames[index].image_path.name,
                "thumbnail_mse": scores[index],
                "second_best_thumbnail_mse": scores[int(ranking[1])],
                "full_resolution_mae_0_255": mae,
                "local_image_sha256": hashlib.sha256(
                    frames[index].image_path.read_bytes()
                ).hexdigest(),
            }
        )
    matches.sort(key=lambda row: row["remote_index"])
    if not matches:
        raise ValueError("No downloaded samples found")
    result = {
        "video": args.video,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "samples": matches,
        "nearest_match_mapping_is_monotonic": all(
            b["nearest_local_index"] > a["nearest_local_index"]
            for a, b in zip(matches, matches[1:])
        ),
        "interpretation": "Nearest-image comparisons are provenance evidence, not byte identity. Nonmonotonic strong matches invalidate assuming the reindexed mirror is chronological. Do not pool local and mirror temporal results.",
    }
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
