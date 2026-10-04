#!/usr/bin/env python3
"""Train the Kalman spatial memory on PolypGen single-frame pseudo-videos.

Data protocol (no positive-sequence frame or label is read):

- positives: ``data_C1``..``data_C6`` single frames, minus
  ``datasets/splits/polypgen/single_frame_test_overlap.txt``;
- absences: ``sequenceData/negativeOnly`` stretches or out-of-view motion;
- test (``scripts/infer.py``): every ``sequenceData/positive`` sequence.

MedSAM2 stays frozen; the model is MedSAM2's ``SAM2Train`` with the Kalman memory
mixin (``training/kalman_trainer.py``). The loss on propagated frames ``t >= 1``
follows MedSAM2's fine-tuning objective (``sam2.1_hiera_tiny_finetune512.yaml``),
with the object-score term split by class:

    present frames:  20 * focal(mask) + 1 * Dice(mask) + 1 * L1(predicted IoU, actual IoU)
                     + 1 * BCE(object score, 1)          # MedSAM2 loss_class weight
    empty frames:    w_absence * BCE(object score, 0)  # --absence_weight, relative to the 1 above
    present frames:  w_distill * ||student memory-conditioned features - native-bank teacher||^2

As in MedSAM2, mask losses are not applied to empty frames, so the object score
is the only absence signal. The final checkpoint is the last step, so no test
label influences model selection.
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
    DIFFICULTIES,
    PseudoVideoGenerator,
    list_negative_sequences,
    list_single_frames,
    load_excluded_paths,
)
from training.kalman_trainer import KalmanLoss, build_training_model, run_clip, video_batch

CHECKPOINT_FORMAT = "adseg_kalman_memory_v2"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/polypgen.yaml")
    parser.add_argument("--polypgen_root", type=Path, default=ROOT / "data/PolypGen2021_MultiCenterData_v3")
    parser.add_argument(
        "--exclude_list", type=Path,
        default=ROOT / "datasets/splits/polypgen/single_frame_test_overlap.txt",
    )
    parser.add_argument("--exclude_negative_sequences", nargs="*", default=[])
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--clip_difficulty", choices=sorted(DIFFICULTIES), default="hard")
    parser.add_argument("--max_frame_gap", type=int, default=None, help="Defaults to the difficulty preset.")
    parser.add_argument("--absence_probability", type=float, default=0.6)
    parser.add_argument("--max_absence", type=int, default=12)
    parser.add_argument("--steps", type=int, default=500, help="Optimizer steps.")
    parser.add_argument("--accumulate", type=int, default=4, help="Clips per optimizer step.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--focal_weight", type=float, default=20.0, help="MedSAM2 loss_mask weight.")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="MedSAM2 loss_dice weight.")
    parser.add_argument("--iou_weight", type=float, default=1.0, help="MedSAM2 loss_iou weight.")
    parser.add_argument(
        "--absence_weight", type=float, default=0.1,
        help="Object-score BCE weight on empty frames, relative to 1.0 on present frames.",
    )
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def frame_stats(student, masks):
    frames, masks = student[1:], masks[1:] > 0
    present = masks.flatten(1).any(1)
    predicted = torch.stack([frame["pred_masks_high_res"][0, 0] > 0 for frame in frames])
    gains = torch.stack([frame["kalman"]["gain"].mean() for frame in frames])
    scores = torch.cat([frame["multistep_object_score_logits"][-1] for frame in frames]).flatten()
    overlap = (predicted & masks).flatten(1).sum(1)
    total = predicted.flatten(1).sum(1) + masks.flatten(1).sum(1)
    mean = lambda values, where: float(values[where].float().mean()) if where.any() else None
    return {
        "dice_present": mean(2 * overlap / total.clamp_min(1), present),
        "false_positive_rate_absent": mean(predicted.flatten(1).any(1), ~present),
        "presence_prob_absent": mean(torch.sigmoid(scores), ~present),
        "gain_present": mean(gains, present),
        "gain_absent": mean(gains, ~present),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    frames = list_single_frames(args.polypgen_root, excluded=load_excluded_paths(args.exclude_list))
    negatives = list_negative_sequences(args.polypgen_root, set(args.exclude_negative_sequences))
    model = build_training_model(config["sam2_cfg"], ROOT / config["sam2_checkpoint"], args.device)
    criterion = KalmanLoss(
        absence_weight=args.absence_weight,
        distill_weight=args.distill_weight,
        focal=args.focal_weight,
        dice=args.dice_weight,
        iou=args.iou_weight,
    )
    generator = PseudoVideoGenerator(
        frames, negatives,
        image_size=model.image_size,
        clip_length=args.clip_length,
        max_frame_gap=args.max_frame_gap,
        absence_probability=args.absence_probability,
        max_absence=args.max_absence,
        difficulty=DIFFICULTIES[args.clip_difficulty],
    )
    optimizer = torch.optim.AdamW(
        model.memory_update.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    run_teacher = args.distill_weight > 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save(path):
        torch.save({
            "format": CHECKPOINT_FORMAT,
            "memory_channels": model.mem_dim,
            "image_channels": model.hidden_dim,
            "update_config": model.memory_update.config,
            "state_dict": model.memory_update.state_dict(),
        }, path)

    setup = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    setup.update({
        "single_frames": len(frames),
        "negative_sequences": len(negatives),
        "positive_sequences_used_for_training": 0,
        "medsam2_frozen": True,
        "checkpoint_selection": "last_step",
    })
    (args.output_dir / "setup.json").write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")

    history = []
    with (args.output_dir / "history.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            clip_stats = []
            for _ in range(args.accumulate):
                clip = generator.sample(rng)
                masks = clip["masks"].to(args.device)
                batch = video_batch(clip["images"].to(args.device), masks)
                student, teacher = run_clip(
                    model, batch, clip["frame_gaps"].cumsum(0).tolist(), run_teacher
                )
                loss, stats = criterion(student, teacher, masks)
                (loss / args.accumulate).backward()
                stats.update(frame_stats(student, masks))
                stats.update({
                    "loss": float(loss),
                    "source": clip["source"],
                    "absence_mode": clip["absence_mode"],
                    "frame_gap": float(clip["frame_gaps"][1]),
                })
                clip_stats.append(stats)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                model.memory_update.parameters(), args.gradient_clip_norm
            ))
            optimizer.step()
            for stats in clip_stats:
                stats.update({"step": step, "gradient_norm": gradient_norm})
                log.write(json.dumps(stats) + "\n")
            log.flush()
            history.extend(clip_stats)
            if step % args.log_every == 0 or step == 1:
                recent = history[-args.log_every * args.accumulate:]
                mean = lambda key: np.nanmean([row[key] if row[key] is not None else np.nan for row in recent])
                print(
                    f"step {step}: loss={mean('loss'):.4f} dice_present={mean('dice_present'):.4f} "
                    f"fp_absent={mean('false_positive_rate_absent'):.3f} "
                    f"gain present/absent={mean('gain_present'):.3f}/{mean('gain_absent'):.3f}",
                    flush=True,
                )
            if step % args.save_every == 0:
                save(args.output_dir / f"kalman_memory_step{step}.pt")
    save(args.output_dir / "kalman_memory.pt")


if __name__ == "__main__":
    main()
