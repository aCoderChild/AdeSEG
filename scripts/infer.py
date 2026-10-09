#!/usr/bin/env python3
"""Run MedSAM2 video inference (native memory bank, or a trained recurrent memory:
RKN or RDE-VOS) from a labelled-video manifest.

The first frame containing each requested label supplies that label's box prompt.
This is an evaluation protocol, not an interactive clinical prompting interface.
For multiple labels, each label is propagated in an independent pass and the
highest positive logit wins where predictions overlap. With ``--nested_labels``,
label k instead stands for the union of all labels >= k (e.g. optic disc = rim +
cup, nasopharynx = airway + adenoid); the passes are painted in increasing label
order, so a lower label keeps only the part not covered by a higher one
(rim = disc - cup).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "external" / "MedSAM2")]

from adenoid.io import load_label_mask
from datasets.video import ManifestFrame, load_manifest
from modeling.detector import PolypDetector
from modeling.recurrent_memory import load_memory_update
from modeling.medsam2 import build_video_predictor

DIAGNOSTIC_FIELDS = ("frame_idx", "frame", "status", "object_score", "presence",
                     "gain_mean", "feature_change_mean", "memory_status",
                     "redetected", "detection_confidence", "mask_box_iou", "rho")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2_cfg", type=Path, required=True)
    parser.add_argument("--sam2_checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--video_ids", nargs="*", default=None)
    parser.add_argument("--label_ids", type=int, nargs="+", default=[1])
    parser.add_argument("--nested_labels", action="store_true",
                        help="Prompt and propagate label k as the union of labels >= k; lower labels keep "
                             "only what higher labels do not cover.")
    parser.add_argument("--memory_checkpoint", type=Path, default=None,
                        help="Trained memory update (RKN or RDE-VOS) from scripts/train_memory.py; "
                             "without it, MedSAM2's native memory bank is used.")
    parser.add_argument("--detector", type=Path, default=Path("checkpoints/polypgen_yolov8n.pt"),
                        help="YOLO weights for re-detection and for a memory update that uses detector observations.")
    parser.add_argument("--redetect", action="store_true",
                        help="Re-detect from the detector box on native MedSAM2 or an update without detector heads.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def first_box(frames: list[ManifestFrame], label_id: int, labels: list[int], nested: bool):
    for index, frame in enumerate(frames):
        mask = load_label_mask(frame.mask_path, labels)
        rows, columns = np.nonzero(mask >= label_id if nested else mask == label_id)
        if rows.size:
            return index, np.array([columns.min(), rows.min(), columns.max(), rows.max()], dtype=np.float32)
    return None, None


def stage_frames(frames: list[ManifestFrame], directory: Path) -> None:
    for index, frame in enumerate(frames):
        (directory / f"{index:06d}.jpg").symlink_to(frame.image_path.resolve())


def retained_memory_mib(output_dict) -> float:
    """MiB of memory state held at the end of a video: every stored frame memory for
    native MedSAM2, the prompt memory and recurrent state (with its covariance) otherwise."""
    if "memory_state" in output_dict:
        tensors = output_dict["memory_state"].tensors()
    else:
        tensors = [
            tensor
            for outputs in (output_dict["cond_frame_outputs"], output_dict["non_cond_frame_outputs"])
            for output in outputs.values()
            for tensor in [output.get("maskmem_features"), *(output.get("maskmem_pos_enc") or [])]
            if tensor is not None
        ]
    unique = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in tensors}
    return sum(unique.values()) / 2 ** 20


def predict_label(predictor, frames: list[ManifestFrame], label_id: int, labels: list[int], recurrent: bool,
                  staged: Path, nested: bool):
    prompt_index, box = first_box(frames, label_id, labels, nested)
    width, height = Image.open(frames[0].image_path).size  # the video's size, as the predictor outputs
    if box is None:
        return np.full((len(frames), height, width), -np.inf, dtype=np.float32), None, {}, {}
    state = predictor.init_state(
        video_path=str(staged),
        offload_video_to_cpu=True,
        offload_state_to_cpu=predictor.device.type == "cuda",
    )
    try:
        if recurrent:
            state.update({"recurrent_memory": True, "memory_anchor_frame_idx": prompt_index})
        if getattr(predictor, "redetect_enabled", False) or getattr(getattr(predictor, "memory_update", None), "uses_detector", False):
            predictor.detections = predictor.detector.frame_detections([f.image_path for f in frames], predictor.device)
        predictor.add_new_points_or_box(state, frame_idx=prompt_index, obj_id=1, box=box)
        logits = np.full((len(frames), state["video_height"], state["video_width"]), -np.inf, dtype=np.float32)
        traces = {}
        started = time.perf_counter()
        for frame_index, _, mask_logits in predictor.propagate_in_video(state, start_frame_idx=prompt_index):
            logits[frame_index] = mask_logits[0, 0].detach().float().cpu().numpy()
            output_type = "cond_frame_outputs" if frame_index == prompt_index else "non_cond_frame_outputs"
            output = state["output_dict"][output_type][frame_index]
            traces[frame_index] = {
                "object_score": float(output["object_score_logits"].float().mean()),
                **output.get("memory_trace", {}),
            }
        elapsed = time.perf_counter() - started
        efficiency = {
            "propagated_frames": len(traces),
            "propagation_seconds": elapsed,
            "frames_per_second": len(traces) / elapsed if elapsed > 0 else float("nan"),
            "retained_memory_mib": retained_memory_mib(state["output_dict"]),
        }
        return logits, prompt_index, traces, efficiency
    finally:
        predictor.reset_state(state)


def write_diagnostics(path: Path, frames: list[ManifestFrame], prompt_index, traces) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for index, frame in enumerate(frames):
            status = "before_prompt" if prompt_index is None or index < prompt_index else (
                "prompt" if index == prompt_index else "propagated"
            )
            writer.writerow({"frame_idx": index, "frame": frame.image_path.stem, "status": status,
                             **traces.get(index, {})})


@torch.inference_mode()
def main():
    args = parse_args()
    recurrent = args.memory_checkpoint is not None
    videos = load_manifest(args.manifest, args.split)
    if args.video_ids is not None:
        videos = {video_id: frames for video_id, frames in videos.items() if video_id in args.video_ids}
        if not videos:
            raise ValueError("None of --video_ids is in the selected split.")
    target = "modeling.recurrent_memory.RecurrentMemoryVideoPredictor" if recurrent or args.redetect else None
    predictor = build_video_predictor(str(args.sam2_cfg), args.sam2_checkpoint, args.device, predictor_target=target)
    if recurrent:
        predictor.memory_update = load_memory_update(args.memory_checkpoint, predictor, predictor.device)
    if args.redetect:
        predictor.redetect_enabled = True
    if args.redetect or getattr(getattr(predictor, "memory_update", None), "uses_detector", False):
        predictor.detector = PolypDetector(args.detector, predictor.device.type)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(args.label_ids) if args.nested_labels else args.label_ids
    run = {
        "manifest": str(args.manifest), "split": args.split, "label_ids": labels,
        "nested_labels": args.nested_labels,
        "memory_update": predictor.memory_update.CHECKPOINT_KIND if recurrent else "native",
        "redetect": bool(args.redetect or getattr(getattr(predictor, "memory_update", None), "uses_detector", False)),
        "memory_checkpoint": str(args.memory_checkpoint) if recurrent else None,
        "device": str(predictor.device), "videos": {},
    }
    for video_id, frames in videos.items():
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="medsam2_frames_") as temporary:
            staged = Path(temporary)
            stage_frames(frames, staged)
            label_logits, prompts, traces, efficiency = {}, {}, {}, {}
            for label_id in labels:
                label_logits[label_id], prompts[label_id], traces[label_id], efficiency[label_id] = predict_label(
                    predictor, frames, label_id, labels, recurrent, staged, args.nested_labels
                )
        height, width = label_logits[labels[0]].shape[-2:]
        merged = np.zeros((len(frames), height, width), dtype=np.uint16 if max(labels) > 255 else np.uint8)
        best = np.full_like(merged, -np.inf, dtype=np.float32)
        for label_id in labels:
            logits = label_logits[label_id]
            take = logits > 0 if args.nested_labels else (logits > 0) & (logits > best)
            merged[take] = label_id
            best[take] = logits[take]
        output = args.output_dir / "masks" / video_id
        output.mkdir(parents=True, exist_ok=True)
        for frame, mask in zip(frames, merged):
            Image.fromarray(mask).save(output / f"{frame.image_path.stem}.png")
        diagnostics = args.output_dir / "masks" / "diagnostics"
        for position, label_id in enumerate(labels):
            # The first label keeps the path evaluation/temporal.py reads; further labels get their own folder.
            folder = diagnostics if position == 0 else diagnostics / f"label{label_id}"
            write_diagnostics(folder / f"{video_id}.csv", frames, prompts[label_id], traces[label_id])
        run["videos"][video_id] = {
            "frames": len(frames), "prompt_frame_index": prompts,
            "elapsed_seconds": time.perf_counter() - started, "efficiency": efficiency,
        }
        print(f"{video_id}: {len(frames)} frames", flush=True)
    (args.output_dir / "setup.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
