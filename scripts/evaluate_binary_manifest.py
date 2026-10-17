#!/usr/bin/env python3
"""Evaluate binary video masks from scripts/infer.py by sequence."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_semantic_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from evaluation.utility_retrieval import temporal_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, groups = [], defaultdict(list)
    for video, frames in load_manifest(args.manifest, args.split).items():
        for frame in frames:
            path = args.pred_dir / "masks" / video / f"{frame.image_path.stem}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            prediction = np.asarray(Image.open(path)) > 0
            target = load_semantic_mask(frame.mask_path) > 0
            scores = calculate_scores(prediction, target)
            row = {"video": video, "frame": frame.frame_index, "present": bool(target.any()), **scores}
            rows.append(row)
            groups[video].append(row)
    per_sequence = {}
    for video, video_rows in groups.items():
        present = [row for row in video_rows if row["present"]]
        absent = [row for row in video_rows if not row["present"]]
        per_sequence[video] = {
            "dice": float(np.mean([row["dice"] for row in present])) if present else None,
            "iou": float(np.mean([row["iou"] for row in present])) if present else None,
            "absent_false_positive_rate": float(np.mean([
                row["specificity"] < 1 for row in absent
            ])) if absent else None,
            **temporal_summary([row["dice"] for row in present]),
        }
    present_sequences = [value for value in per_sequence.values() if value["dice"] is not None]
    summary = {
        "sequences": len(per_sequence),
        "dice": float(np.mean([value["dice"] for value in present_sequences])),
        "iou": float(np.mean([value["iou"] for value in present_sequences])),
        "failure_rate": float(np.mean([value["failure_rate"] for value in present_sequences])),
        "mean_episode_length": float(np.mean([value["mean_episode_length"] for value in present_sequences])),
        "recovery_rate": float(np.mean([value["recovery_rate"] for value in present_sequences])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "per_sequence": per_sequence, "frames": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
