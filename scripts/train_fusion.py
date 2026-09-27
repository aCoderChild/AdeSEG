#!/usr/bin/env python3
"""Archived PolypGen overfit script for the compact-memory ablation."""

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
from training.dynamic_state_trainer import DynamicStateVideoTrainer
from training.losses import dice_bce_loss
from training.video_trainer import fusion_optimizer


def box_from_mask(mask):
    y, x = torch.where(mask > 0)
    return torch.tensor([x.min(), y.min(), x.max(), y.max()], dtype=torch.float32)


def load_clips(root, sequence, clip_length, transform, clips_per_sequence, seed):
    all_samples = list(iter_polypgen_samples(root, sequence))
    valid_starts = [
        start
        for start in range(len(all_samples) - clip_length + 1)
        if all_samples[start].masks["polyp"].any()
    ]
    if not valid_starts:
        return []
    rng = random.Random(f"{seed}:{sequence}")
    starts = rng.sample(valid_starts, k=min(clips_per_sequence, len(valid_starts)))
    clips = []
    for start in sorted(starts):
        samples = all_samples[start : start + clip_length]
        images = torch.stack([transform(Image.open(sample.image_path).convert("RGB")) for sample in samples])
        masks = torch.stack([torch.from_numpy(sample.masks["polyp"]).float() for sample in samples])
        masks = F.interpolate(masks.unsqueeze(1), size=images.shape[-2:], mode="nearest").squeeze(1)
        clips.append({
            "sequence": sequence,
            "start": start,
            "frames": [sample.image_path.name for sample in samples],
            "images": images.unsqueeze(0),
            "masks": masks.unsqueeze(0),
            "boxes": box_from_mask(masks[0]).unsqueeze(0),
        })
    return clips


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=ROOT / "configs/polypgen.yaml")
parser.add_argument("--sequences", nargs="+", default=["seq2", "seq3"])
parser.add_argument("--clip_length", type=int, default=4)
parser.add_argument("--clips_per_sequence", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--device", default="mps")
parser.add_argument("--initial_fusion_checkpoint", type=Path, default=None)
parser.add_argument("--output_dir", type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text())
trainer = DynamicStateVideoTrainer(config["sam2_cfg"], ROOT / config["sam2_checkpoint"], args.device)
if args.initial_fusion_checkpoint is not None:
    try:
        initial_state = torch.load(args.initial_fusion_checkpoint, map_location=args.device, weights_only=True)
    except TypeError:
        initial_state = torch.load(args.initial_fusion_checkpoint, map_location=args.device)
    trainer.state_fusion.load_state_dict(initial_state)
optimizer = fusion_optimizer(trainer, lr=1e-4)
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
for step in range(args.steps):
    clip = clips[step % len(clips)]
    images, masks, boxes = clip["images"], clip["masks"], clip["boxes"]
    images, masks, boxes = images.to(args.device), masks.to(args.device), boxes.to(args.device)
    _, logits, _ = trainer(images, boxes, labels=None)
    prediction = torch.stack(logits, dim=1).squeeze(2)
    loss = dice_bce_loss(prediction[:, 1:], masks[:, 1:])
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    dice = ((prediction[:, 1:].sigmoid() > 0.5) & (masks[:, 1:] > 0)).sum().float() * 2
    dice /= (prediction[:, 1:].sigmoid() > 0.5).sum() + (masks[:, 1:] > 0).sum()
    gate = trainer.state_fusion.last_gate
    gradient_norm = torch.sqrt(sum((parameter.grad.square().sum() for parameter in trainer.state_fusion.parameters() if parameter.grad is not None), torch.zeros((), device=args.device)))
    history.append({
        "step": step + 1, "sequence": clip["sequence"], "clip_start": clip["start"],
        "loss": float(loss.detach()), "dice": float(dice.detach()),
        "fusion_gradient_norm": float(gradient_norm.detach()),
        "gate_mean": float(gate.mean()), "gate_min": float(gate.min()), "gate_max": float(gate.max()),
    })
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "setup.json").write_text(json.dumps({
    "sequences": args.sequences,
    "clip_length": args.clip_length,
    "clips_per_sequence": args.clips_per_sequence,
    "seed": args.seed,
    "initial_fusion_checkpoint": str(args.initial_fusion_checkpoint) if args.initial_fusion_checkpoint else None,
    "selected_clips": [
        {key: clip[key] for key in ("sequence", "start", "frames")} for clip in clips
    ],
    "sequences_without_usable_clips": [
        sequence for sequence in args.sequences
        if not any(clip["sequence"] == sequence for clip in clips)
    ],
}, indent=2) + "\n")
(args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
torch.save(trainer.state_fusion.state_dict(), args.output_dir / "fusion.pt")
summary = {
    "first": history[0], "last": history[-1],
    "fusion_gradient_nonzero": any(parameter.grad is not None and bool(parameter.grad.abs().sum()) for parameter in trainer.state_fusion.parameters()),
    "fusion_parameters_changed": any(not torch.equal(before, after) for before, after in zip(initial_parameters, trainer.state_fusion.parameters())),
}
(args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
