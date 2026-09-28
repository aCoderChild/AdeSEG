#!/usr/bin/env python3
"""Tiny-overfit training for the reliability-gated memory experiment.

This script intentionally trains only ``ReliabilityGatedFusion``.  MedSAM2 is
frozen and in evaluation mode; propagated predictions, never ground-truth
masks, are encoded into the recurrent state.  Ground-truth masks supervise the
future-frame Dice+BCE loss only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MedSAM2")]

from datasets.polypgen import iter_polypgen_samples
from training.losses import dice_bce_loss
from training.rgm_trainer import ReliabilityGatedMemoryTrainer, rgm_optimizer


def box_from_mask(mask: torch.Tensor) -> torch.Tensor:
    y, x = torch.where(mask > 0)
    if not len(x):
        raise ValueError("The prompt frame must contain foreground pixels.")
    return torch.tensor([x.min(), y.min(), x.max(), y.max()], dtype=torch.float32)


def load_clips(data_root, sequence, clip_length, transform, clips_per_sequence, seed):
    samples = list(iter_polypgen_samples(data_root, sequence))
    valid_starts = [
        start for start in range(len(samples) - clip_length + 1)
        if samples[start].masks["polyp"].any()
    ]
    rng = random.Random(f"{seed}:{sequence}")
    starts = rng.sample(valid_starts, k=min(clips_per_sequence, len(valid_starts)))
    clips = []
    for start in sorted(starts):
        window = samples[start : start + clip_length]
        images = torch.stack([transform(Image.open(sample.image_path).convert("RGB")) for sample in window])
        masks = torch.stack([torch.from_numpy(sample.masks["polyp"]).float() for sample in window])
        masks = F.interpolate(masks.unsqueeze(1), size=images.shape[-2:], mode="nearest").squeeze(1)
        clips.append({
            "sequence": sequence,
            "start": start,
            "frames": [sample.image_path.name for sample in window],
            "images": images.unsqueeze(0),
            "masks": masks.unsqueeze(0),
            "boxes": box_from_mask(masks[0]).unsqueeze(0),
        })
    return clips


def dice_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    prediction = logits.sigmoid() > 0.5
    target = targets > 0
    numerator = 2.0 * (prediction & target).sum(dtype=torch.float32)
    denominator = prediction.sum(dtype=torch.float32) + target.sum(dtype=torch.float32)
    return numerator / denominator.clamp_min(1.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/polypgen.yaml")
    parser.add_argument("--sequences", nargs="+", default=["seq2", "seq3"])
    parser.add_argument("--clip_length", type=int, default=4)
    parser.add_argument("--clips_per_sequence", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cpu")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.clip_length < 2:
        raise ValueError("--clip_length must be at least 2.")
    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    trainer = ReliabilityGatedMemoryTrainer(
        config["sam2_cfg"], ROOT / config["sam2_checkpoint"], args.device,
    )
    optimizer = rgm_optimizer(trainer, lr=args.learning_rate)
    # eval disables frozen MedSAM2 dropout without disabling gradients through it.
    trainer.model.eval()
    trainer.state_fusion.train()
    initial_parameters = [parameter.detach().clone() for parameter in trainer.state_fusion.parameters()]
    clips = [
        clip
        for sequence in args.sequences
        for clip in load_clips(
            ROOT / config["data_root"], sequence, args.clip_length,
            trainer._transforms, args.clips_per_sequence, args.seed,
        )
    ]
    if not clips:
        raise ValueError("No supervised clips were selected.")

    history = []
    frozen_model_has_gradients = False
    for step in range(args.steps):
        clip = clips[step % len(clips)]
        images = clip["images"].to(args.device)
        masks = clip["masks"].to(args.device)
        boxes = clip["boxes"].to(args.device)
        _, logits, _ = trainer(images, boxes, labels=None)
        prediction = torch.stack(logits, dim=1).squeeze(2)
        loss = dice_bce_loss(prediction[:, 1:], masks[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        frozen_model_has_gradients |= any(
            parameter.grad is not None and bool(parameter.grad.abs().sum())
            for parameter in trainer.model.parameters()
        )
        gradient_norm = torch.sqrt(sum(
            (parameter.grad.square().sum() for parameter in trainer.state_fusion.parameters() if parameter.grad is not None),
            torch.zeros((), device=args.device),
        ))
        optimizer.step()
        gate = trainer.state_fusion.last_gate
        history.append({
            "step": step + 1,
            "sequence": clip["sequence"],
            "clip_start": clip["start"],
            "loss": float(loss.detach()),
            "dice": float(dice_score(prediction[:, 1:], masks[:, 1:]).detach()),
            "fusion_gradient_norm": float(gradient_norm.detach()),
            "gate_mean": float(gate.mean()),
            "gate_min": float(gate.min()),
            "gate_max": float(gate.max()),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "adseg_rgm_rde_livos_v1",
        "feature_channels": trainer.model.mem_dim,
        "state_dict": trainer.state_fusion.state_dict(),
    }
    torch.save(checkpoint, args.output_dir / "rgm_fusion.pt")
    setup = {
        "sequences": args.sequences,
        "clip_length": args.clip_length,
        "clips_per_sequence": args.clips_per_sequence,
        "seed": args.seed,
        "device": args.device,
        "learning_rate": args.learning_rate,
        "medsam2_frozen": True,
        "memory_update_uses": "predicted_masks",
        "loss_supervision": "future_frame_dice_bce",
        "selected_clips": [
            {key: clip[key] for key in ("sequence", "start", "frames")} for clip in clips
        ],
    }
    (args.output_dir / "setup.json").write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    summary = {
        "first": history[0],
        "last": history[-1],
        "loss_decreased": history[-1]["loss"] < history[0]["loss"],
        "fusion_gradient_nonzero": any(item["fusion_gradient_norm"] > 0.0 for item in history),
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
