#!/usr/bin/env python3
"""Train a memory update (the modified RKN with detector observations, or RDE-VOS with its
official objective) on labelled videos, with MedSAM2 frozen.
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
sys.path[:0] = [str(ROOT), str(ROOT / "external" / "MedSAM2")]

from adenoid.io import load_label_mask
from datasets.video import ManifestVideoClips, assert_disjoint_videos, load_manifest
from modeling.detector import PolypDetector
from modeling.recurrent_memory import save_memory_update
from modeling.medsam2 import build_video_predictor
from scripts.infer import predict_label, stage_frames
from training.memory_trainer import RDELoss, RKNLoss, build_training_model, run_clip, video_batch
from training.protocol import load_protocol, verify_training
from training.tracking import finish, log, start_wandb

# Adam. Modified RKN: the RKN notebook's lr 5e-4 decayed to 5e-5 and weight decay 1e-4. RDE-VOS:
# official weight decay 1e-7 (its lr 1e-5 is for 150k iterations, so 1e-4 for these 500 steps).
DEFAULTS = {
    "rkn": {"learning_rate": 5e-4, "final_learning_rate": 5e-5, "weight_decay": 1e-4},
    "rde": {"learning_rate": 1e-4, "final_learning_rate": None, "weight_decay": 1e-7},
}


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
    parser.add_argument("--protocol", type=Path, default=None,
                        help="Optional JSON protocol; when supplied, enforce its exact training and validation videos.")
    parser.add_argument("--memory_update", choices=("rkn", "rde"), required=True,
                        help="Update rule to train: the modified RKN or RDE-VOS.")
    parser.add_argument("--label_ids", type=int, nargs="+", default=[1], help="Semantic-mask IDs; e.g. 1 for binary or 1 2 for adenoid + airway.")
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--fp_weight", type=float, default=0.5,
                        help="Checkpoint selection (and the RKN presence loss) maximise polyp-frame Dice - fp_weight x "
                             "empty-frame FP rate on the held-out videos.")
    parser.add_argument("--sample_by_video", action=argparse.BooleanOptionalAction, default=True,
                        help="Draw a video uniformly, then one of its clips (default), so long videos do not dominate.")
    parser.add_argument("--steps", type=int, default=500, help="Optimizer steps.")
    parser.add_argument("--accumulate", type=int, default=4, help="Clips per optimizer step.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Default: the rule's official value.")
    parser.add_argument("--final_learning_rate", type=float, default=None,
                        help="Exponential decay to this lr over --steps (RKN notebook: 5e-5).")
    parser.add_argument("--weight_decay", type=float, default=None, help="Default: the rule's official value.")
    rkn = parser.add_argument_group("modified RKN")
    rkn.add_argument("--rkn_weight_factor", type=float, default=0.1,
                     help="Scale of the RKN networks' initial weights (RKN notebook: 0.1).")
    rkn.add_argument("--detector", type=Path, default=ROOT / "checkpoints/polypgen_yolov8n.pt",
                     help="YOLO weights (trained without sequenceData/positive frames).")
    rkn.add_argument("--presence_only", action="store_true",
                     help="Ablation: learn only the presence correction (fixed Q, R, rho; no presence-scaled R or pointer gating).")
    rkn.add_argument("--presence_weight", type=float, default=1.0)
    rkn.add_argument("--reliability_weight", type=float, default=0.5)
    rkn.add_argument("--nll_weight", type=float, default=0.02)
    rkn.add_argument("--absence_fraction", type=float, default=0.0,
                     help="Share of training clips drawn from clips that contain an absent frame.")
    rde = parser.add_argument_group("RDE-VOS (official objective and memory mode)")
    rde.add_argument("--rde_mode", choices=("two-frames-compress", "gt-compress"), default="two-frames-compress",
                     help="Official memory-bank mode, in training and inference.")
    rde.add_argument("--mem_every", type=int, default=3, help="Official: compress into the RDE every N frames.")
    rde.add_argument("--rde_repeat", type=int, default=0, help="Official SAM `repeat` (SE blocks).")
    rde.add_argument("--rde_distill_weight", type=float, default=10.0, help="Official decoder_f2/f4_weight.")
    rde.add_argument("--temperature", type=float, default=1.0, help="Official KL temperature.")
    rde.add_argument("--top_p", type=float, default=0.15, help="Official bootstrapped-CE pixel fraction.")
    rde.add_argument("--start_warm", type=int, default=None,
                     help="Bootstrapping starts (default: 20k/150k of --steps, the official stage-3 ratio).")
    rde.add_argument("--end_warm", type=int, default=None, help="Bootstrapping reaches top_p (default: 70k/150k of --steps).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--wandb_project", default="ade-seg")
    parser.add_argument("--wandb_entity", default="adenoid-hypertrophy")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb_base_url", default="https://api.wandb.ai")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def sequence_scores(predictor, videos, label_ids, recurrent=True):
    """Inference-path scores on whole videos (as scripts/infer.py, prompt on the first
    labelled frame), averaged over sequences: polyp-frame Dice, empty-frame FP rate, lost
    polyp frames (Dice < 0.1), and the RKN's mean gain on polyp and on empty frames."""
    present_dice, absent_fp, lost, gains = [], [], [], {True: [], False: []}
    for frames in videos.values():
        with tempfile.TemporaryDirectory(prefix="memory_val_") as temporary:
            stage_frames(frames, Path(temporary))
            for label_id in label_ids:
                logits, prompt, traces, _ = predict_label(predictor, frames, label_id, label_ids, recurrent,
                                                          Path(temporary), False)
                if prompt is None:
                    continue
                dice, empty_fp = [], []
                for index in range(prompt + 1, len(frames)):
                    frame, frame_logits = frames[index], logits[index]
                    predicted, target = frame_logits > 0, load_label_mask(frame.mask_path, label_ids) == label_id
                    gain = traces.get(index, {}).get("gain_mean", float("nan"))
                    if not np.isnan(gain):
                        gains[bool(target.any())].append(gain)
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
        "gain_present": float(np.mean(gains[True])) if gains[True] else None,
        "gain_absent": float(np.mean(gains[False])) if gains[False] else None,
    }


@torch.no_grad()
def frame_stats(student, masks):
    frames, masks = student[1:], masks[1:] > 0
    present = masks.flatten(1).any(1)
    predicted = torch.stack([frame["pred_masks_high_res"][0, 0] > 0 for frame in frames])
    gains = [frame["memory"].get("gain") for frame in frames]  # only the RKN has a gain
    gains = torch.stack([gain.mean() for gain in gains]) if all(g is not None for g in gains) else None
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
        "gain_present": mean(gains, present) if gains is not None else None,
        "gain_absent": mean(gains, ~present) if gains is not None else None,
    }


def main():
    args = parse_args()
    if args.manifest.resolve() == (ROOT / "data" / "polypgen_sequence.jsonl").resolve() and args.protocol is None:
        raise ValueError("PolypGen training requires --protocol to enforce its video-level partition.")
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
    protocol = load_protocol(args.protocol) if args.protocol else None
    if protocol:
        verify_training(protocol, train_videos, validation_videos)
    is_rde = args.memory_update == "rde"
    update_kwargs = ({"mode": args.rde_mode, "mem_every": args.mem_every, "repeat": args.rde_repeat} if is_rde
                     else {"weight_factor": args.rkn_weight_factor, "learn_memory": not args.presence_only})
    model = build_training_model(str(args.sam2_cfg), args.sam2_checkpoint, args.device, args.memory_update,
                                 **update_kwargs)
    args.start_warm = round(args.steps * 20 / 150) if args.start_warm is None else args.start_warm
    args.end_warm = round(args.steps * 70 / 150) if args.end_warm is None else args.end_warm
    criterion = RDELoss(
        distill_weight=args.rde_distill_weight, temperature=args.temperature,
        top_p=args.top_p, start_warm=args.start_warm, end_warm=args.end_warm,
    ) if is_rde else RKNLoss(args.presence_weight, 0.0 if args.presence_only else args.reliability_weight,
                             0.0 if args.presence_only else args.nll_weight, args.fp_weight)
    for key, value in DEFAULTS[args.memory_update].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    train_clips = ManifestVideoClips(
        train_videos, args.label_ids, model.image_size, args.clip_length, sample_by_video=args.sample_by_video,
        absence_fraction=0.0 if is_rde else args.absence_fraction,
    )
    # Validation runs whole held-out videos through the inference predictor, sharing the
    # trained module, so selection sees the drift that 16-frame clips hide.
    predictor = build_video_predictor(
        str(args.sam2_cfg), args.sam2_checkpoint, args.device,
        predictor_target="modeling.recurrent_memory.RecurrentMemoryVideoPredictor",
    )
    predictor.memory_update = model.memory_update
    detector = None
    if getattr(model.memory_update, "uses_detector", False):
        detector = PolypDetector(args.detector, args.device)
        detector([frame.image_path for frames in (*train_videos.values(), *validation_videos.values()) for frame in frames])
        predictor.detector = detector

    def validate():
        return sequence_scores(predictor, validation_videos, args.label_ids)

    optimizer = torch.optim.Adam([p for p in model.memory_update.parameters() if p.requires_grad],
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.final_learning_rate is not None:  # Trainer.set_decreasing_learning_rate, step_size 1
        gamma = (args.final_learning_rate / args.learning_rate) ** (1 / args.steps)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    native_teacher = is_rde and args.rde_distill_weight > 0
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
        "checkpoint_selection": f"max held-out per-sequence polyp-frame Dice - {args.fp_weight} x empty-frame FP rate",
        "teacher": "native MedSAM2 bank" if native_teacher else None,
        "protocol_name": protocol.get("name") if protocol else None,
        "objective": ("official RDE-VOS (bootstrapped CE + KL distillation)" if is_rde
                      else "modified RKN: later frames' mask loss through MedSAM2 + cost-weighted presence BCE + "
                           "soft-IoU reliability BCE + official GaussianLikelihoodLoss of the carried GT memory"),
    })
    (args.output_dir / "setup.json").write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")
    wandb_run = start_wandb(args.wandb_project, args.wandb_entity, args.wandb_name, args.wandb_mode,
                            args.wandb_base_url, setup, args.output_dir, (args.memory_update, "training"))

    history = []
    def selection_score(scores):
        return scores["present_dice"] - args.fp_weight * (scores["absent_fp_rate"] or 0.0)

    best = {"step": 0, **validate()}
    save(args.output_dir / "memory_update.pt")
    validation_log = [best]
    print(f"step 0 validation: {best}", flush=True)
    with (args.output_dir / "history.jsonl").open("w", encoding="utf-8") as history_file:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            clip_stats = []
            for _ in range(args.accumulate):
                clip = train_clips.sample(rng)
                masks = clip["masks"].to(args.device)
                batch = video_batch(clip["images"].to(args.device), masks)
                detections = detector.frame_detections(clip["image_paths"], args.device) if detector else None
                student, teacher = run_clip(model, batch, native_teacher, detections)
                loss, stats = criterion(student, teacher, masks, step)
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
            gradient_norm = float(torch.nn.utils.get_total_norm(
                [p.grad for p in model.memory_update.parameters() if p.grad is not None]))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for stats in clip_stats:
                stats.update({"step": step, "gradient_norm": gradient_norm})
                history_file.write(json.dumps(stats) + "\n")
            history_file.flush()
            history.extend(clip_stats)
            step_mean = lambda key: np.nanmean([row[key] if row[key] is not None else np.nan for row in clip_stats])
            log(wandb_run, "train", {
                "loss": step_mean("loss"), "dice_present": step_mean("dice_present"),
                "fp_absent": step_mean("false_positive_rate_absent"),
                "closed_present": step_mean("closed_gate_present"),
                "gain_present": step_mean("gain_present"), "gain_absent": step_mean("gain_absent"),
                "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": gradient_norm,
            }, step)
            if step % args.log_every == 0 or step == 1:
                recent = history[-args.log_every * args.accumulate:]
                mean = lambda key: np.nanmean([row[key] if row[key] is not None else np.nan for row in recent])
                train_metrics = {
                    "loss": mean("loss"), "dice_present": mean("dice_present"),
                    "fp_absent": mean("false_positive_rate_absent"), "closed_present": mean("closed_gate_present"),
                    "gain_present": mean("gain_present"), "gain_absent": mean("gain_absent"),
                    "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": gradient_norm,
                }
                print(
                    f"step {step}: loss={train_metrics['loss']:.4f} dice_present={train_metrics['dice_present']:.4f} "
                    f"fp_absent={train_metrics['fp_absent']:.3f} closed_present={train_metrics['closed_present']:.3f} "
                    f"gain present/absent={train_metrics['gain_present']:.3f}/{train_metrics['gain_absent']:.3f}",
                    flush=True,
                )
            if step % args.save_every == 0:
                save(args.output_dir / f"memory_update_step{step}.pt")
                scores = {"step": step, **validate()}
                validation_log.append(scores)
                print(f"step {step} validation: {scores}", flush=True)
                log(wandb_run, "validation", scores, step)
                if selection_score(scores) > selection_score(best):
                    best = scores
                    save(args.output_dir / "memory_update.pt")
    save(args.output_dir / "memory_update_last.pt")
    (args.output_dir / "validation.json").write_text(
        json.dumps({"selected": best, "history": validation_log}, indent=2) + "\n", encoding="utf-8"
    )
    log(wandb_run, "best_validation", best, best["step"])
    finish(wandb_run)


if __name__ == "__main__":
    main()
