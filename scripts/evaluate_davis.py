#!/usr/bin/env python3
"""Evaluate indexed DAVIS predictions with official region J and boundary F."""

from __future__ import annotations

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
from evaluation.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaluation.utility_retrieval import temporal_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Directory containing masks/<video>/<frame>.png from scripts/infer.py.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    videos = load_manifest(args.manifest, args.split)
    rows = []
    by_object = defaultdict(list)
    for video, frames in videos.items():
        object_ids = sorted(int(value) for value in np.unique(load_semantic_mask(frames[0].mask_path)) if value)
        for frame in frames:
            ground_truth = load_semantic_mask(frame.mask_path)
            path = args.pred_dir / "masks" / video / f"{frame.image_path.stem}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            prediction = np.asarray(Image.open(path))
            for object_id in object_ids:
                j = float(db_eval_iou(ground_truth == object_id, prediction == object_id))
                boundary_f = float(db_eval_boundary(ground_truth == object_id, prediction == object_id))
                row = {"video": video, "object": object_id, "frame": frame.frame_index,
                       "J": j, "F": boundary_f, "J&F": (j + boundary_f) / 2}
                rows.append(row)
                by_object[f"{video}:{object_id}"].append(row)
    per_object = {}
    for name, object_rows in by_object.items():
        per_object[name] = {
            "J": float(np.mean([row["J"] for row in object_rows])),
            "F": float(np.mean([row["F"] for row in object_rows])),
            "J&F": float(np.mean([row["J&F"] for row in object_rows])),
            **temporal_summary([row["J"] for row in object_rows]),
        }
    summary = {
        "sequence_object_pairs": len(per_object),
        "J": float(np.mean([value["J"] for value in per_object.values()])),
        "F": float(np.mean([value["F"] for value in per_object.values()])),
        "J&F": float(np.mean([value["J&F"] for value in per_object.values()])),
        "failure_rate": float(np.mean([value["failure_rate"] for value in per_object.values()])),
        "mean_episode_length": float(np.mean([value["mean_episode_length"] for value in per_object.values()])),
        "recovery_rate": float(np.mean([value["recovery_rate"] for value in per_object.values()])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "per_object": per_object, "frames": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
