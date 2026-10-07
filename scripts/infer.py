#!/usr/bin/env python3
"""Run native or Kalman MedSAM2 video inference from a labelled-video manifest.

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
sys.path[:0] = [str(ROOT), str(ROOT / "MedSAM2")]

from adenoid.io import load_semantic_mask
from datasets.video import ManifestFrame, load_manifest
from modeling.ema_memory import ConstantGainMemory
from modeling.kalman_memory import KalmanMemoryUpdate, load_memory_update
from modeling.medsam2 import build_video_predictor

DIAGNOSTIC_FIELDS = ("frame_idx", "frame", "status", "object_score", "presence",
                     "gain_mean", "feature_change_mean", "kalman_status")


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
    parser.add_argument("--memory_backend", choices=("native", "kalman"), default="kalman")
    parser.add_argument("--kalman_checkpoint", type=Path, default=None)
    parser.add_argument("--untrained", action="store_true",
                        help="Use a freshly initialised Kalman update instead of a checkpoint. Its last noise "
                             "layer and uncertainty embedding are zero, so it is the hand-set filter for any seed.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--skip_absent", action="store_true",
                        help="Treat a frame MedSAM2 calls empty as a missing measurement (K=0).")
    parser.add_argument("--fixed_gain", type=float, nargs=2, default=None, metavar=("PRESENT", "ABSENT"),
                        help="EMA baseline (modeling/ema_memory.py): constant-gain memory instead of the "
                             "Kalman update, with these present / absent gains. No checkpoint needed.")
    return parser.parse_args()


def label_region(path: Path, label_id: int, nested: bool) -> np.ndarray:
    mask = load_semantic_mask(path)
    return mask >= label_id if nested else mask == label_id


def first_box(frames: list[ManifestFrame], label_id: int, nested: bool) -> tuple[int | None, np.ndarray | None]:
    for index, frame in enumerate(frames):
        rows, columns = np.nonzero(label_region(frame.mask_path, label_id, nested))
        if rows.size:
            return index, np.array([columns.min(), rows.min(), columns.max(), rows.max()], dtype=np.float32)
    return None, None


def stage_frames(frames: list[ManifestFrame], directory: Path) -> None:
    for index, frame in enumerate(frames):
        (directory / f"{index:06d}.jpg").symlink_to(frame.image_path.resolve())


def retained_memory_mib(output_dict) -> float:
    """MiB of memory state held at the end of a video: every stored frame memory for
    native MedSAM2, the anchor/position/mean/variance tensors for the 1-slot memory."""
    if "kalman_state" in output_dict:
        tensors = output_dict["kalman_state"].tensors()
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


def predict_label(predictor, frames: list[ManifestFrame], label_id: int, backend: str, staged: Path, nested: bool):
    prompt_index, box = first_box(frames, label_id, nested)
    height, width = load_semantic_mask(frames[0].mask_path).shape
    if box is None:
        return np.full((len(frames), height, width), -np.inf, dtype=np.float32), None, {}, {}
    state = predictor.init_state(
        video_path=str(staged),
        offload_video_to_cpu=True,
        offload_state_to_cpu=predictor.device.type == "cuda",
    )
    try:
        if backend == "kalman":
            state.update({"kalman_enabled": True, "kalman_anchor_frame_idx": prompt_index})
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
                **output.get("kalman_trace", {}),
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
    if args.memory_backend == "kalman" and sum((args.kalman_checkpoint is not None, args.untrained,
                                                args.fixed_gain is not None)) != 1:
        raise ValueError("--memory_backend kalman needs exactly one of --kalman_checkpoint, --untrained, --fixed_gain.")
    videos = load_manifest(args.manifest, args.split)
    if args.video_ids is not None:
        videos = {video_id: frames for video_id, frames in videos.items() if video_id in args.video_ids}
        if not videos:
            raise ValueError("None of --video_ids is in the selected split.")
    target = "modeling.kalman_memory.KalmanMemoryVideoPredictor" if args.memory_backend == "kalman" else None
    predictor = build_video_predictor(str(args.sam2_cfg), args.sam2_checkpoint, args.device, predictor_target=target)
    if args.memory_backend == "kalman" and args.fixed_gain is not None:
        # EMA baseline: constant-gain memory, no Kalman adaptivity (see modeling/ema_memory.py).
        predictor.memory_update = ConstantGainMemory(
            predictor.mem_dim, predictor.hidden_dim, args.fixed_gain[0], args.fixed_gain[1])
    elif args.memory_backend == "kalman" and args.untrained:
        predictor.memory_update = KalmanMemoryUpdate(predictor.mem_dim, predictor.hidden_dim).to(predictor.device).eval()
    elif args.memory_backend == "kalman":
        predictor.memory_update = load_memory_update(args.kalman_checkpoint, predictor, predictor.device)
    if args.memory_backend == "kalman":
        predictor.kalman_skip_absent = args.skip_absent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(args.label_ids) if args.nested_labels else args.label_ids
    run = {
        "manifest": str(args.manifest), "split": args.split, "label_ids": labels,
        "nested_labels": args.nested_labels, "memory_backend": args.memory_backend,
        "kalman_checkpoint": str(args.kalman_checkpoint) if args.kalman_checkpoint else None,
        "untrained": args.untrained, "fixed_gain": args.fixed_gain, "skip_absent": args.skip_absent,
        "device": str(predictor.device), "videos": {},
    }
    for video_id, frames in videos.items():
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="kalman_frames_") as temporary:
            staged = Path(temporary)
            stage_frames(frames, staged)
            label_logits, prompts, traces, efficiency = {}, {}, {}, {}
            for label_id in labels:
                label_logits[label_id], prompts[label_id], traces[label_id], efficiency[label_id] = predict_label(
                    predictor, frames, label_id, args.memory_backend, staged, args.nested_labels
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
