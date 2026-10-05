#!/usr/bin/env python3
"""Calibrate the Kalman innovation gate on held-out training pseudo-videos (no test video).

Runs the Kalman memory (gate off, optional absence skip) on validation clips built
from the frames ``train_kalman.py`` holds out (every 10th single frame and negative
sequence), and records, for every propagated polyp frame whose presence gate is
open, the cosine distance between its object prototype and the last accepted one,
with the frame's IoU. The gate is the given percentile of that distance over
correctly tracked frames (IoU >= 0.5): at most (100 - percentile)% of good frames
are rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MedSAM2")]

from datasets.pseudo_video import (
    DIFFICULTIES, PseudoVideoGenerator, list_negative_sequences, list_single_frames, load_excluded_paths,
)
from modeling.kalman_memory import load_memory_update
from training.kalman_trainer import build_training_model, run_clip, video_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/polypgen.yaml")
    parser.add_argument("--polypgen_root", type=Path, default=ROOT / "data/PolypGen2021_MultiCenterData_v3")
    parser.add_argument("--exclude_list", type=Path, default=ROOT / "datasets/splits/polypgen/single_frame_test_overlap.txt")
    parser.add_argument("--kalman_checkpoint", type=Path, required=True)
    parser.add_argument("--skip_absent", action="store_true")
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True, help="JSON with the gate and the distance statistics.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    frames = list_single_frames(args.polypgen_root, excluded=load_excluded_paths(args.exclude_list))[::10]
    negatives = list_negative_sequences(args.polypgen_root)[::10]
    model = build_training_model(config["sam2_cfg"], ROOT / config["sam2_checkpoint"], args.device)
    model.memory_update = load_memory_update(args.kalman_checkpoint, model, args.device)
    model.kalman_skip_absent = args.skip_absent
    generator = PseudoVideoGenerator(frames, negatives, image_size=model.image_size, difficulty=DIFFICULTIES["hard"])
    rng = np.random.default_rng(args.seed)

    distances, ious = [], []
    with torch.no_grad():
        for _ in range(args.clips):
            clip = generator.sample(rng)
            masks = clip["masks"].to(args.device)
            student, _ = run_clip(model, video_batch(clip["images"].to(args.device), masks), False)
            for frame, target in zip(student[1:], masks[1:] > 0):
                kalman = frame.get("kalman")
                if kalman is None or not target.any() or float(kalman["presence"].mean()) <= 0.5:
                    continue
                predicted = frame["pred_masks_high_res"][0, 0] > 0
                union = (predicted | target).sum()
                ious.append(float((predicted & target).sum() / union) if union else 0.0)
                distances.append(float(kalman["distance"].mean()))
    distances, ious = np.array(distances), np.array(ious)
    good, wrong = distances[ious >= 0.5], distances[ious < 0.1]
    gate = float(np.percentile(good, args.percentile))
    report = {
        "gate": gate,
        "percentile": args.percentile,
        "skip_absent": args.skip_absent,
        "frames": len(distances),
        "good_frames": len(good),
        "wrong_frames": len(wrong),
        "good_distance_median": float(np.median(good)),
        "wrong_distance_median": float(np.median(wrong)) if len(wrong) else None,
        "wrong_frames_rejected": float((wrong > gate).mean()) if len(wrong) else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
