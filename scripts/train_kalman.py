#!/usr/bin/env python3
"""Train Kalman memory update from any labelled video dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MedSAM2")]

from adenoid.io import load_label_mask
from datasets.video import ManifestVideoClips, assert_disjoint_videos, load_manifest
from modeling.kalman_memory import save_memory_update
from modeling.medsam2 import build_video_predictor
from scripts.infer import predict_label, stage_frames
from training.kalman_trainer import EMA_GAINS, KalmanLoss, build_training_model, ema_teacher, run_clip, video_batch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sam2_cfg", type=Path, required=True)
    parser.add_argument("--sam2_checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL frame manifest; paths are relative to it.")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--val_videos", nargs="*", default=None,
                        help="Validate on these videos of --train_split instead of --val_split "
                             "(e.g. PolypGen: --train_split dev --val_videos seq13 seq14 seq15).")
    parser.add_argument("--memory_update", choices=("kalman", "rde"), default="kalman",
                        help="Update rule to train: the Kalman update or the RDE-VOS aggregation module.")
    parser.add_argument("--label_ids", type=int, nargs="+", default=[1], help="Semantic-mask IDs; e.g. 1 for binary or 1 2 for adenoid + airway.")
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--sample_by_video", action=argparse.BooleanOptionalAction, default=True,
                        help="Draw a video uniformly, then one of its clips (default), so long videos do not dominate.")
    parser.add_argument("--steps", type=int, default=500, help="Optimizer steps.")
    parser.add_argument("--accumulate", type=int, default=4, help="Clips per optimizer step.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--focal_weight", type=float, default=20.0, help="MedSAM2 loss_mask weight.")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="MedSAM2 loss_dice weight.")
    parser.add_argument(
        "--iou_weight", type=float, default=0.0,
        help="MedSAM2 loss_iou weight; off because the IoU head is frozen and not an output.",
    )
    parser.add_argument(
        "--absence_weight", type=float, default=1.0,
        help="Object-score BCE weight on empty frames, relative to 1.0 on present frames.",
    )
    parser.add_argument(
        "--distill_weight", type=float, default=1.0,
        help="Feature MSE to the EMA teacher on present frames. Kalman and RDE share this and every other default.",
    )
    parser.add_argument("--no_worse_weight", type=float, default=1.0,
                        help="Penalty where the student is worse than the EMA teacher on a present frame.")
    parser.add_argument("--teacher_gains", type=float, nargs=2, default=list(EMA_GAINS), metavar=("PRESENT", "ABSENT"),
                        help="EMA teacher gains (the EMA baseline's).")
    parser.add_argument("--anchor_weight", type=float, default=1.0,
                        help="L2 pull of the update's parameters toward their initial (EMA-like) values.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def sequence_scores(predictor, videos, label_ids):
    """Inference-path scores on whole videos (as scripts/infer.py, prompt on the first
    labelled frame), averaged over sequences: polyp-frame Dice, empty-frame FP rate and
    lost polyp frames (Dice < 0.1)."""
    present_dice, absent_fp, lost = [], [], []
    for frames in videos.values():
        with tempfile.TemporaryDirectory(prefix="kalman_val_") as temporary:
            stage_frames(frames, Path(temporary))
            for label_id in label_ids:
                logits, prompt, _, _ = predict_label(predictor, frames, label_id, label_ids, "kalman", Path(temporary), False)
                if prompt is None:
                    continue
                dice, empty_fp = [], []
                for frame, frame_logits in zip(frames[prompt + 1:], logits[prompt + 1:]):
                    predicted, target = frame_logits > 0, load_label_mask(frame.mask_path, label_ids) == label_id
                    if target.any():
                        dice.append(2 * (predicted & target).sum() / (predicted.sum() + target.sum()))
                    else:
                        empty_fp.append(predicted.any())
                if dice:
                    present_dice.append(np.mean(dice))
                    lost.append(np.mean(np.array(dice) < 0.1))
                if empty_fp:
                    absent_fp.append(np.mean(empty_fp))
    return {
        "present_dice": float(np.mean(present_dice)),
        "absent_fp_rate": float(np.mean(absent_fp)) if absent_fp else None,
        "lost": float(np.mean(lost)),
    }


@torch.no_grad()
def frame_stats(student, masks):
    frames, masks = student[1:], masks[1:] > 0
    present = masks.flatten(1).any(1)
    predicted = torch.stack([frame["pred_masks_high_res"][0, 0] > 0 for frame in frames])
    gains = torch.stack([frame["kalman"]["gain"].mean() for frame in frames])
    closed = torch.cat([frame["multistep_object_score_logits"][-1] for frame in frames]).flatten() <= 0
    scores = torch.cat([frame["multistep_object_score_logits"][-1] for frame in frames]).flatten()
    overlap = (predicted & masks).flatten(1).sum(1)
    total = predicted.flatten(1).sum(1) + masks.flatten(1).sum(1)
    mean = lambda values, where: float(values[where].float().mean()) if where.any() else None
    return {
        "dice_present": mean(2 * overlap / total.clamp_min(1), present),
        "false_positive_rate_absent": mean(predicted.flatten(1).any(1), ~present),
        "presence_prob_absent": mean(torch.sigmoid(scores), ~present),
        "closed_gate_present": mean(closed, present),
        "gain_present": mean(gains, present),
        "gain_absent": mean(gains, ~present),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_videos = load_manifest(args.manifest, args.train_split)
    if args.val_videos:
        missing = sorted(set(args.val_videos) - set(train_videos))
        if missing:
            raise ValueError(f"--val_videos not in split {args.train_split!r}: {missing}")
        validation_videos = {v: f for v, f in train_videos.items() if v in args.val_videos}
        train_videos = {v: f for v, f in train_videos.items() if v not in args.val_videos}
    else:
        validation_videos = load_manifest(args.manifest, args.val_split)
    assert_disjoint_videos(train_videos, validation_videos)
    model = build_training_model(str(args.sam2_cfg), args.sam2_checkpoint, args.device, args.memory_update)
    criterion = KalmanLoss(
        absence_weight=args.absence_weight,
        distill_weight=args.distill_weight,
        no_worse_weight=args.no_worse_weight,
        focal=args.focal_weight,
        dice=args.dice_weight,
        iou=args.iou_weight,
    )
    train_clips = ManifestVideoClips(
        train_videos, args.label_ids, model.image_size, args.clip_length, sample_by_video=args.sample_by_video
    )
    # Validation runs whole held-out videos through the inference predictor, sharing the
    # trained module, so selection sees the drift that 16-frame clips hide.
    predictor = build_video_predictor(
        str(args.sam2_cfg), args.sam2_checkpoint, args.device,
        predictor_target="modeling.kalman_memory.KalmanMemoryVideoPredictor",
    )
    predictor.memory_update = model.memory_update

    def validate():
        return sequence_scores(predictor, validation_videos, args.label_ids)

    parameters = [parameter for parameter in model.memory_update.parameters() if parameter.requires_grad]
    initial = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    teacher_update = ema_teacher(model, args.teacher_gains) if args.distill_weight > 0 or args.no_worse_weight > 0 else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save(path):
        save_memory_update(model.memory_update, path)

    setup = {key: json.loads(json.dumps(value, default=str)) for key, value in vars(args).items()}
    setup.update({
        "train_videos": len(train_videos),
        "validation_videos": len(validation_videos),
        "train_promptable_clips": len(train_clips),
        "validation": "whole held-out videos, inference path",
        "medsam2_frozen": True,
        "checkpoint_selection": "max held-out per-sequence polyp-frame Dice with empty-frame FP <= step 0; else step 0",
        "teacher": "EMA",
    })
    (args.output_dir / "setup.json").write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")

    history = []
    best = {"step": 0, **validate()}
    fp_limit = best["absent_fp_rate"]
    save(args.output_dir / "kalman_memory.pt")
    validation_log = [best]
    print(f"step 0 validation: {best}", flush=True)
    with (args.output_dir / "history.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            clip_stats = []
            for _ in range(args.accumulate):
                clip = train_clips.sample(rng)
                masks = clip["masks"].to(args.device)
                batch = video_batch(clip["images"].to(args.device), masks)
                student, teacher = run_clip(model, batch, teacher_update)
                loss, stats = criterion(student, teacher, masks)
                if loss.requires_grad:  # false when no propagated frame contributes to the loss
                    (loss / args.accumulate).backward()
                stats.update(frame_stats(student, masks))
                stats.update({
                    "loss": float(loss.detach()),
                    "source": clip["source"],
                    "video_id": clip["video_id"],
                    "label_id": clip["label_id"],
                    "absence_mode": clip["absence_mode"],
                    "frame_gap": float(clip["frame_gaps"][1]),
                })
                clip_stats.append(stats)
            anchor = sum(((parameter - start) ** 2).sum() for parameter, start in zip(parameters, initial))
            if args.anchor_weight > 0:
                (args.anchor_weight * anchor).backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip_norm))
            optimizer.step()
            for stats in clip_stats:
                stats.update({"step": step, "gradient_norm": gradient_norm, "anchor": float(anchor.detach())})
                log.write(json.dumps(stats) + "\n")
            log.flush()
            history.extend(clip_stats)
            if step % args.log_every == 0 or step == 1:
                recent = history[-args.log_every * args.accumulate:]
                mean = lambda key: np.nanmean([row[key] if row[key] is not None else np.nan for row in recent])
                print(
                    f"step {step}: loss={mean('loss'):.4f} dice_present={mean('dice_present'):.4f} "
                    f"fp_absent={mean('false_positive_rate_absent'):.3f} "
                    f"closed_present={mean('closed_gate_present'):.3f} "
                    f"gain present/absent={mean('gain_present'):.3f}/{mean('gain_absent'):.3f}",
                    flush=True,
                )
            if step % args.save_every == 0:
                save(args.output_dir / f"kalman_memory_step{step}.pt")
                scores = {"step": step, **validate()}
                validation_log.append(scores)
                print(f"step {step} validation: {scores}", flush=True)
                fp_ok = fp_limit is None or scores["absent_fp_rate"] <= fp_limit
                if fp_ok and scores["present_dice"] > best["present_dice"]:
                    best = scores
                    save(args.output_dir / "kalman_memory.pt")
    save(args.output_dir / "kalman_memory_last.pt")
    (args.output_dir / "validation.json").write_text(
        json.dumps({"selected": best, "history": validation_log}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
