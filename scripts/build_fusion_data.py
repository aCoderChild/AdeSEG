#!/usr/bin/env python3
"""Presence-fusion training data from cross-fitted pseudo-videos (training data only).

For each fold k, clips are generated from the single frames and negative
sequences held out of detector k (``build_detector_dataset.py --holdout_fold k``),
so the detections are out-of-fold. Each clip runs through the Kalman predictor
with detector observations, exactly as at test time, and every propagated frame
is logged with its presence features, label and whether it had a confident
detection (the fusion head is fitted on the frames without one).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "MedSAM2"):
    sys.path.insert(0, str(path))

from datasets.pseudo_video import (
    DIFFICULTIES, IMAGE_MEAN, IMAGE_STD, PseudoVideoGenerator,
    list_negative_sequences, list_single_frames, load_excluded_paths,
)
from modeling.detector_observation import PRESENCE_FEATURES, top_detections
from modeling.kalman_memory import load_memory_update
from modeling.medsam2 import build_video_predictor, load_yolo_model


def write_clip(clip, directory: Path) -> list[float]:
    times = clip["frame_gaps"].cumsum(0).tolist()
    pixels = clip["images"].permute(0, 2, 3, 1).numpy() * IMAGE_STD + IMAGE_MEAN
    for time, image in zip(times, pixels):
        rgb = (image.clip(0, 1) * 255).round().astype(np.uint8)
        cv2.imwrite(str(directory / f"{int(time):05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return times


def first_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/polypgen.yaml")
    parser.add_argument("--polypgen_root", type=Path, default=ROOT / "data/PolypGen2021_MultiCenterData_v3")
    parser.add_argument("--exclude_list", type=Path, default=ROOT / "datasets/splits/polypgen/single_frame_test_overlap.txt")
    parser.add_argument("--detectors", type=Path, nargs="+", required=True, help="best.pt of fold 0, 1, ...")
    parser.add_argument("--memory_backend", choices=["kalman", "native"], default="kalman")
    parser.add_argument("--kalman_checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="CSV of per-frame features and labels.")
    parser.add_argument("--clips_per_fold", type=int, default=300)
    parser.add_argument("--clip_difficulty", choices=sorted(DIFFICULTIES), default="hard")
    parser.add_argument("--observation_conf", type=float, default=0.5)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    folds = len(args.detectors)

    kalman = args.memory_backend == "kalman"
    predictor = build_video_predictor(
        config["sam2_cfg"], ROOT / config["sam2_checkpoint"], args.device,
        predictor_target="modeling.detector_observation." + ("ObservedKalmanVideoPredictor" if kalman else "ObservedVideoPredictor"),
    )
    if kalman:
        predictor.memory_update = load_memory_update(args.kalman_checkpoint, predictor, args.device)
    predictor.observation_conf = args.observation_conf

    frames = list_single_frames(args.polypgen_root, excluded=load_excluded_paths(args.exclude_list))
    sequences = list_negative_sequences(args.polypgen_root)
    rng = np.random.default_rng(args.seed)
    fields = ["fold", "clip", "frame", "present", "observed", "absence_mode", *PRESENCE_FEATURES]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle, torch.inference_mode():
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fold, weights in enumerate(args.detectors):
            detector = load_yolo_model(weights)
            generator = PseudoVideoGenerator(
                [f for f in frames if (int(f.center[1:]) - 1) % folds == fold],
                [seq for i, seq in enumerate(sequences) if i % folds == fold],
                difficulty=DIFFICULTIES[args.clip_difficulty],
            )
            for clip_index in range(args.clips_per_fold):
                clip = generator.sample(rng)
                with tempfile.TemporaryDirectory() as directory:
                    times = write_clip(clip, Path(directory))
                    names = sorted(Path(directory).glob("*.jpg"))
                    observations = dict(enumerate(top_detections(detector, names[1:]), start=1))
                    state = predictor.init_state(video_path=directory, offload_video_to_cpu=True)
                if kalman:
                    state.update({"kalman_enabled": True, "kalman_anchor_frame_idx": 0})
                predictor.observations = observations
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=first_box(clip["masks"][0].numpy()))
                for frame_idx, _, _ in predictor.propagate_in_video(state, start_frame_idx=0):
                    trace = state["output_dict"]["non_cond_frame_outputs"].get(frame_idx, {}).get("presence_trace")
                    if trace is None:
                        continue
                    writer.writerow({
                        "fold": fold, "clip": clip_index, "frame": frame_idx,
                        "present": int(clip["present"][frame_idx]), "observed": trace["observed"],
                        "absence_mode": clip["absence_mode"],
                        **{name: trace[f"feature_{name}"] for name in PRESENCE_FEATURES},
                    })
                print(f"fold {fold} clip {clip_index + 1}/{args.clips_per_fold}", flush=True)


if __name__ == "__main__":
    main()
