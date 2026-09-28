#!/usr/bin/env python3
"""Create reproducible YOLO box prompts for PolypGen sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.polypgen import get_frame_names, get_video_frame_dir, resolve_frame_path
from modeling.medsam2 import get_yolo_boxes, load_yolo_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--yolo_checkpoint", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be positive.")
    yolo = load_yolo_model(args.yolo_checkpoint)
    records = {}
    missing = []
    for sequence in args.sequences:
        frame_dir = get_video_frame_dir(args.data_root, sequence)
        frames = get_frame_names(frame_dir)
        selected = None
        # Match scripts/infer.py: freeze the first accepted detection as the
        # one prompt from which propagation starts.
        for frame_idx, frame_name in enumerate(frames):
            if frame_idx % args.stride:
                continue
            boxes = get_yolo_boxes(
                yolo, resolve_frame_path(frame_dir, frame_name),
                image_size=args.image_size, confidence=args.confidence, max_boxes=1,
            )
            if boxes:
                box, score = boxes[0]
                selected = {
                    "frame_idx": frame_idx,
                    "frame": frame_name,
                    "box": [float(value) for value in box],
                    "confidence": score,
                }
                break
        if selected is None:
            missing.append(sequence)
            continue
        records[sequence] = selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    summary = {
        "requested_sequences": args.sequences,
        "prompted_sequences": sorted(records),
        "missing_sequences": missing,
        "confidence": args.confidence,
        "image_size": args.image_size,
        "stride": args.stride,
    }
    args.output.with_name(f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
