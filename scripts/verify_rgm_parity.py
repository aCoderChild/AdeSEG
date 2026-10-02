#!/usr/bin/env python3
"""Compare frozen RGM training and inference trajectories on one fixed clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MedSAM2")]

from datasets.polypgen import get_frame_names, get_video_frame_dir, resolve_frame_path
from modeling.medsam2 import build_video_predictor, load_video_frame_like_predictor
from modeling.rgm_memory import ReliabilityGatedFusion
from training.rgm_trainer import ReliabilityGatedMemoryTrainer


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tensor_error(training: torch.Tensor, inference: torch.Tensor) -> dict:
    training = training.float()
    inference = inference.float()
    resized = False
    if training.shape != inference.shape:
        if training.ndim != 4 or inference.ndim != 4:
            return {"shape_match": False, "training_shape": list(training.shape), "inference_shape": list(inference.shape)}
        inference = F.interpolate(inference, size=training.shape[-2:], mode="bilinear", align_corners=False)
        resized = True
    difference = (training - inference).abs()
    return {
        "shape_match": True,
        "resized_mask_logits": resized,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
    }


def _trace_errors(training_trace, inference_trace) -> list[dict]:
    if len(training_trace) != len(inference_trace):
        raise RuntimeError(f"Trace length mismatch: train={len(training_trace)}, inference={len(inference_trace)}.")
    rows = []
    for train_frame, infer_frame in zip(training_trace, inference_trace):
        if train_frame["frame_idx"] != infer_frame["frame_idx"]:
            raise RuntimeError("Frame indices differ between training and inference traces.")
        row = {"frame_idx": train_frame["frame_idx"]}
        for name in ("mask_logits", "predicted_iou", "object_pointer", "candidate", "state"):
            row[name] = _tensor_error(train_frame[name], infer_frame[name])
        if train_frame["gate"] is None or infer_frame["gate"] is None:
            row["gate"] = {"both_none": train_frame["gate"] is None and infer_frame["gate"] is None}
        else:
            row["gate"] = _tensor_error(train_frame["gate"], infer_frame["gate"])
        rows.append(row)
    return rows


def _passed(rows, tolerance: float) -> bool:
    for row in rows:
        for name, result in row.items():
            if name in {"frame_idx", "gate"} and isinstance(result, dict) and "both_none" in result:
                continue
            if isinstance(result, dict) and (not result.get("shape_match", True) or result.get("max_abs_error", 0.0) > tolerance):
                return False
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/polypgen.yaml")
    parser.add_argument("--prompt_records", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--clip_length", type=int, default=4)
    parser.add_argument("--fixed_gate", type=float, default=None)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cpu")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.02)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.clip_length < 2:
        raise ValueError("--clip_length must be at least two frames.")
    config = _load_config(args.config)
    records = json.loads(args.prompt_records.read_text(encoding="utf-8"))
    prompt = records.get(args.sequence)
    if prompt is None:
        raise KeyError(f"No saved prompt for {args.sequence}.")
    video_dir = get_video_frame_dir(ROOT / config["data_root"], args.sequence)
    names = get_frame_names(video_dir)
    prompt_idx = int(prompt["frame_idx"])
    frame_names = names[prompt_idx : prompt_idx + args.clip_length]
    if len(frame_names) != args.clip_length:
        raise ValueError("The requested clip extends past the sequence end.")

    trainer = ReliabilityGatedMemoryTrainer(
        config["sam2_cfg"], ROOT / config["sam2_checkpoint"], args.device,
        fixed_gate=args.fixed_gate,
    )
    trainer.model.eval()
    trainer.state_fusion.eval()
    images = torch.stack([
        load_video_frame_like_predictor(resolve_frame_path(video_dir, name), trainer.model.image_size)
        for name in frame_names
    ]).unsqueeze(0).to(args.device)
    original = Image.open(resolve_frame_path(video_dir, frame_names[0]))
    width, height = original.size
    scale = torch.tensor([trainer.model.image_size / width, trainer.model.image_size / height] * 2)
    model_box = torch.tensor(prompt["box"], dtype=torch.float32) * scale
    with torch.no_grad():
        trainer(
            images,
            model_box.unsqueeze(0).to(args.device),
            labels=None,
            video_num_frames=len(names),
        )
    train_trace = trainer.frame_trace

    predictor = build_video_predictor(
        config["sam2_cfg"], ROOT / config["sam2_checkpoint"], device=args.device,
        predictor_target="modeling.rgm_memory.ReliabilityGatedMemoryVideoPredictor",
    )
    predictor.eval()
    predictor.state_fusion = ReliabilityGatedFusion(
        predictor.mem_dim, fixed_gate=args.fixed_gate
    ).to(args.device)
    predictor.state_fusion.load_state_dict(trainer.state_fusion.state_dict())
    predictor.state_fusion.eval()
    inference_state = predictor.init_state(
        video_path=video_dir, offload_video_to_cpu=False, offload_state_to_cpu=False
    )
    inference_state.update({
        "rgm_enabled": True,
        "rgm_anchor_frame_idx": prompt_idx,
        "rgm_capture_tensors": True,
    })
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=prompt_idx,
        obj_id=1,
        box=torch.tensor(prompt["box"], dtype=torch.float32).numpy(),
    )
    with torch.no_grad():
        list(predictor.propagate_in_video(inference_state))
    raw_trace = inference_state["output_dict"].get("rgm_frame_trace", [])
    inference_trace = [
        item for item in raw_trace if prompt_idx <= item["frame_idx"] < prompt_idx + args.clip_length
    ]
    # Convert absolute video indices to the clip indices used by the trainer.
    for item in inference_trace:
        item["frame_idx"] -= prompt_idx
    rows = _trace_errors(train_trace, inference_trace)
    summary = {
        "sequence": args.sequence,
        "prompt_frame_idx": prompt_idx,
        "prompt_box": prompt["box"],
        "frame_names": frame_names,
        "device": args.device,
        "tolerance": args.tolerance,
        "passed": _passed(rows, args.tolerance),
        "frames": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "parity.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit("RGM training/inference parity check failed.")


if __name__ == "__main__":
    main()
