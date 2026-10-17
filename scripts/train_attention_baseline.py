#!/usr/bin/env python3
"""Train the matched-budget attention baseline with segmentation loss only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "external" / "MedSAM2")]

from datasets.video import ManifestVideoClips, assert_disjoint_videos, load_manifest
from adenoid.io import load_semantic_mask
from modeling.medsam2 import build_video_predictor
from modeling.utility_memory import attach_utility_controller, save_utility_head
from training.tracking import finish, log, start_wandb
from training.utility_trainer import (
    build_attention_baseline, run_attention_clip, segmentation_loss,
    sequence_scores, video_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-cfg", required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="holdout")
    parser.add_argument("--label-ids", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--wandb-project", default="ade-seg")
    parser.add_argument("--wandb-entity", default="adenoid-hypertrophy")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-base-url", default="https://api.wandb.ai")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_videos = load_manifest(args.manifest, args.train_split)
    validation_videos = load_manifest(args.manifest, args.validation_split)
    assert_disjoint_videos(train_videos, validation_videos)
    if args.label_ids is None:
        args.label_ids = sorted({
            int(value)
            for frames in (*train_videos.values(), *validation_videos.values())
            for value in np.unique(load_semantic_mask(frames[0].mask_path))
            if value
        })
    model, controller = build_attention_baseline(
        args.sam2_cfg, args.sam2_checkpoint, args.device
    )
    clips = ManifestVideoClips(
        train_videos, args.label_ids, model.image_size, args.clip_length,
        sample_by_video=True,
    )
    predictor = build_video_predictor(args.sam2_cfg, args.sam2_checkpoint, args.device)
    attach_utility_controller(predictor, controller)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.learning_rate)
    setup = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    setup.update({
        "objective": "segmentation focal+Dice on self-predicted rollouts; no utility labels",
        "trainable_parameters": sum(parameter.numel() for parameter in controller.parameters()),
        "backbone_frozen": True,
        "prompt_type": "ground-truth mask",
    })
    (args.output_dir / "setup.json").write_text(json.dumps(setup, indent=2) + "\n")
    run = start_wandb(args.wandb_project, args.wandb_entity, args.wandb_name,
                      args.wandb_mode, args.wandb_base_url, setup, args.output_dir,
                      ("attention-baseline", "segmentation-loss", f"seed-{args.seed}"))
    best = {"step": -1, "present_dice": -float("inf")}
    try:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(args.accumulate):
                clip = clips.sample(rng)
                masks = clip["masks"].to(args.device)
                outputs = run_attention_clip(
                    model, video_batch(clip["images"].to(args.device), masks)
                )
                loss = segmentation_loss(outputs, masks)
                (loss / args.accumulate).backward()
                losses.append(float(loss.detach()))
            optimizer.step()
            log(run, "train", {"loss": float(np.mean(losses)),
                                "bias_scale": float(controller.bias_scale.detach())}, step)
            if step % args.validate_every == 0 or step == args.steps:
                controller.eval()
                scores = sequence_scores(
                    predictor, validation_videos, args.label_ids,
                )
                controller.train()
                log(run, "validation", scores, step)
                if scores["present_dice"] > best["present_dice"]:
                    best = {"step": step, **scores}
                    save_utility_head(controller, args.output_dir / "attention_baseline.pt", best)
                print(f"step {step}: {scores}", flush=True)
        (args.output_dir / "validation.json").write_text(json.dumps(best, indent=2) + "\n")
    finally:
        finish(run)


if __name__ == "__main__":
    main()
