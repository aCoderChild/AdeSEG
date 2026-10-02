#!/usr/bin/env python3
"""Train frozen-MedSAM2 RGM from reproducible YOLO box prompts."""

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

from datasets.polypgen import iter_polypgen_samples
from modeling.medsam2 import load_video_frame_like_predictor
from training.losses import dice_bce_loss
from training.rgm_trainer import ReliabilityGatedMemoryTrainer, rgm_optimizer


# seq7 has no frozen YOLO prompt and is intentionally absent from prompt_records.
DEFAULT_TRAIN_SEQUENCES = (
    "seq2", "seq3", "seq4", "seq5", "seq6", "seq8", "seq9", "seq10",
    "seq11", "seq12", "seq13", "seq14", "seq15",
)


def load_prompted_clip(data_root, sequence, prompt, clip_length, image_size):
    """Load the prompt frame and its following frames using inference preprocessing."""
    samples = list(iter_polypgen_samples(data_root, sequence))
    start = int(prompt["frame_idx"])
    window = samples[start : start + clip_length]
    if len(window) != clip_length:
        raise ValueError(f"{sequence}: prompt does not leave {clip_length} frames.")
    if window[0].image_path.stem != prompt["frame"]:
        raise ValueError(f"{sequence}: prompt frame does not match dataset ordering.")
    images = torch.stack([
        load_video_frame_like_predictor(sample.image_path, image_size) for sample in window
    ])
    masks = torch.stack([torch.from_numpy(sample.masks["polyp"]).float() for sample in window])
    masks = F.interpolate(masks.unsqueeze(1), size=images.shape[-2:], mode="nearest").squeeze(1)
    width, height = Image.open(window[0].image_path).size
    scale = torch.tensor([image_size / width, image_size / height] * 2)
    return {
        "sequence": sequence,
        "start": start,
        "frames": [sample.image_path.name for sample in window],
        "images": images.unsqueeze(0),
        "masks": masks.unsqueeze(0),
        "boxes": (torch.tensor(prompt["box"], dtype=torch.float32) * scale).unsqueeze(0),
        "video_num_frames": len(samples),
    }


def dice_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    prediction = logits.sigmoid() > 0.5
    target = targets > 0
    numerator = 2.0 * (prediction & target).sum(dtype=torch.float32)
    denominator = prediction.sum(dtype=torch.float32) + target.sum(dtype=torch.float32)
    return float(numerator / denominator.clamp_min(1.0))


def trace_diagnostics(trace, targets):
    rows = []
    for item in trace:
        frame_idx = item["frame_idx"]
        logits = item["mask_logits"]
        target = targets[:, frame_idx : frame_idx + 1].cpu()
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
        previous = item["previous_state"]
        rows.append({
            "frame_idx": frame_idx,
            "predicted_iou": float(item["predicted_iou"].mean()),
            "frame_dice": dice_score(logits, target),
            "gate": None if item["gate"] is None else float(item["gate"].mean()),
            "candidate_previous_distance": None if previous is None else float(
                (item["candidate"] - previous).norm()
            ),
            "state_norm": float(item["state"].norm()),
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/polypgen.yaml")
    parser.add_argument("--prompt_records", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_TRAIN_SEQUENCES))
    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=("gate", "joint"), default="gate")
    parser.add_argument(
        "--fixed_gate",
        type=float,
        default=None,
        help="Keep the update gate fixed while training the Conv3D fusion only.",
    )
    parser.add_argument("--gate_learning_rate", type=float, default=1e-4)
    parser.add_argument("--fusion_learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cpu")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.clip_length < 2 or args.steps < 1:
        raise ValueError("clip_length must be at least two and steps must be positive.")
    if args.fixed_gate is not None and not 0.0 < args.fixed_gate < 1.0:
        raise ValueError("--fixed_gate must be in (0, 1).")
    if args.fixed_gate is not None and args.stage != "joint":
        raise ValueError("--fixed_gate requires --stage joint to train the Conv3D fusion.")
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prompt_records = json.loads(args.prompt_records.read_text(encoding="utf-8"))
    missing = [sequence for sequence in args.sequences if sequence not in prompt_records]
    if missing:
        raise KeyError(f"Missing frozen prompts for: {', '.join(missing)}")
    trainer = ReliabilityGatedMemoryTrainer(
        config["sam2_cfg"], ROOT / config["sam2_checkpoint"], args.device,
        fixed_gate=args.fixed_gate,
    )
    # Frozen modules use inference behavior, while the selected RGM layers train.
    trainer.model.eval()
    trainer.state_fusion.train()
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device, weights_only=True)
        if checkpoint.get("format") != "adseg_rgm_rde_livos_v1":
            raise ValueError("resume_checkpoint has an unrecognized RGM format.")
        if checkpoint.get("feature_channels") != trainer.model.mem_dim:
            raise ValueError("resume_checkpoint has an incompatible feature width.")
        saved_fixed_gate = checkpoint.get("fusion_config", {}).get("fixed_gate")
        if saved_fixed_gate != args.fixed_gate:
            raise ValueError("resume_checkpoint fixed-gate setting does not match this run.")
        trainer.state_fusion.load_state_dict(checkpoint["state_dict"])
    optimizer = rgm_optimizer(
        trainer,
        gate_lr=args.gate_learning_rate,
        fusion_lr=args.fusion_learning_rate,
        stage=args.stage,
    )
    clips = [
        load_prompted_clip(
            ROOT / config["data_root"], sequence, prompt_records[sequence],
            args.clip_length, trainer.model.image_size,
        )
        for sequence in args.sequences
    ]
    initial_parameters = [parameter.detach().clone() for parameter in trainer.state_fusion.parameters()]
    history = []
    frozen_model_has_gradients = False
    for step in range(args.steps):
        clip = clips[step % len(clips)]
        images = clip["images"].to(args.device)
        masks = clip["masks"].to(args.device)
        boxes = clip["boxes"].to(args.device)
        _, logits, _ = trainer(
            images, boxes, labels=None, video_num_frames=clip["video_num_frames"]
        )
        prediction = torch.stack(logits, dim=1).squeeze(2)
        loss = dice_bce_loss(prediction[:, 1:], masks[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        frozen_model_has_gradients |= any(
            parameter.grad is not None and bool(parameter.grad.abs().sum())
            for parameter in trainer.model.parameters()
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in trainer.state_fusion.parameters() if parameter.requires_grad],
            args.gradient_clip_norm,
        )
        optimizer.step()
        history.append({
            "step": step + 1,
            "sequence": clip["sequence"],
            "clip_start": clip["start"],
            "loss": float(loss.detach()),
            "future_frame_dice": dice_score(prediction[:, 1:], masks[:, 1:]),
            "fusion_gradient_norm_before_clip": float(gradient_norm),
            "frames": trace_diagnostics(trainer.frame_trace, masks),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "adseg_rgm_rde_livos_v1",
        "feature_channels": trainer.model.mem_dim,
        "fusion_config": {
            "initial_gate": trainer.state_fusion.initial_gate,
            "fixed_gate": trainer.state_fusion.fixed_gate,
        },
        "state_dict": trainer.state_fusion.state_dict(),
    }, args.output_dir / "rgm_fusion.pt")
    setup = {
        "sequences": args.sequences,
        "clip_length": args.clip_length,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        "stage": args.stage,
        "fixed_gate": args.fixed_gate,
        "gate_learning_rate": args.gate_learning_rate,
        "fusion_learning_rate": args.fusion_learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "resume_checkpoint": None if args.resume_checkpoint is None else str(args.resume_checkpoint),
        "medsam2_frozen_eval": True,
        "memory_update_uses": "predicted_masks",
        "loss_supervision": "future_frame_dice_bce",
        "prompt_records": str(args.prompt_records),
        "selected_clips": [
            {key: clip[key] for key in ("sequence", "start", "frames", "video_num_frames")}
            for clip in clips
        ],
    }
    (args.output_dir / "setup.json").write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    per_sequence = {}
    for sequence in args.sequences:
        sequence_history = [row for row in history if row["sequence"] == sequence]
        first, last = sequence_history[0], sequence_history[-1]
        per_sequence[sequence] = {
            "first_loss": first["loss"],
            "last_loss": last["loss"],
            "loss_delta": last["loss"] - first["loss"],
            "first_future_frame_dice": first["future_frame_dice"],
            "last_future_frame_dice": last["future_frame_dice"],
            "future_frame_dice_delta": last["future_frame_dice"] - first["future_frame_dice"],
        }
    summary = {
        "first": history[0],
        "last": history[-1],
        "per_sequence": per_sequence,
        "fusion_gradient_nonzero": any(item["fusion_gradient_norm_before_clip"] > 0.0 for item in history),
        "fusion_parameters_changed": any(
            not torch.equal(before, after)
            for before, after in zip(initial_parameters, trainer.state_fusion.parameters())
        ),
        "frozen_medsam2_has_gradients": frozen_model_has_gradients,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
