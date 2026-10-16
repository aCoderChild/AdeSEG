#!/usr/bin/env python3
"""Generate frame-level leave-one-memory-out utility labels on cached rollouts.

This is the shared E1/E2/E4 instrumentation path.  It first makes an untouched
baseline rollout, identifies preregistered failure-episode onsets, then replays
the sequence and evaluates exclusions at those frames.  Image features and the
memory state are reused by ``SAM2VideoPredictor``; object-pointer tokens are
never masked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import cv2
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "external" / "MedSAM2")]

from adenoid.io import load_label_mask, load_semantic_mask
from datasets.video import load_manifest
from evaluation.utility_retrieval import failure_episodes, rank_agreement
from modeling.medsam2 import build_video_predictor
from modeling.utility_memory import UtilityController, UtilityHead, attach_utility_controller
from scripts.infer import predict_label, stage_frames
from training.tracking import finish, log, start_wandb


def save_progress(output_dir, records, rows, *, state, video=None, label_id=None):
    """Persist completed work so an interrupted Kaggle stage remains recoverable."""
    output_dir = Path(output_dir)
    records_tmp = output_dir / "utility_records.pt.tmp"
    torch.save(records, records_tmp)
    os.replace(records_tmp, output_dir / "utility_records.pt")

    rows_tmp = output_dir / "frames.jsonl.tmp"
    with rows_tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(rows_tmp, output_dir / "frames.jsonl")

    status = {
        "state": state,
        "completed_rows": len(rows),
        "utility_units": len(records),
        "last_video": video,
        "last_label_id": label_id,
        "updated_unix": time.time(),
    }
    status_tmp = output_dir / "status.json.tmp"
    status_tmp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(status_tmp, output_dir / "status.json")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-cfg", required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--label-ids", type=int, nargs="*", default=None,
                        help="Objects to analyze. Default: all nonzero IDs in each video's first mask.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--max-analyzed-frames", type=int, default=None)
    parser.add_argument("--frame-selection", choices=("failure_onsets", "all", "every_n"),
                        default="failure_onsets")
    parser.add_argument("--every-n", type=int, default=5)
    parser.add_argument("--random-null", action=argparse.BooleanOptionalAction, default=True,
                        help="Evaluate the E2 best-of-N matched random control.")
    parser.add_argument("--memory-source", choices=("predicted", "ground_truth", "corrupted"),
                        default="predicted")
    parser.add_argument("--corruption-level", type=float, default=0.0)
    parser.add_argument("--corruption-type", choices=("erosion", "dilation", "shift", "blob", "deletion", "mixed"),
                        default="mixed")
    parser.add_argument("--wandb-project", default="ade-seg")
    parser.add_argument("--wandb-entity", default="adenoid-hypertrophy")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-base-url", default="https://api.wandb.ai")
    return parser.parse_args()


def dice(prediction, target):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    denominator = prediction.sum() + target.sum()
    return float(2 * (prediction & target).sum() / denominator) if denominator else 1.0


def _mask_at_original_size(logits, shape):
    resized = F.interpolate(logits.float(), size=shape, mode="bilinear", align_corners=False)
    return resized[0, 0].detach().cpu().numpy() > 0


def _attention_by_unit(trace):
    units, layers = trace["units"], trace["attention"]
    if not layers:
        return [float("nan")] * len(units)
    values = []
    for unit in units:
        per_layer = [
            weights[..., unit["start"] : unit["end"]].sum(dim=-1).mean().item()
            for weights in layers
        ]
        values.append(float(np.mean(per_layer)))
    return values


def _json_unit(unit):
    return {
        key: value for key, value in unit.items()
        if key not in {"predicted_iou", "object_score"}
    } | {
        "predicted_iou": float(unit["predicted_iou"].float().mean())
        if unit.get("predicted_iou") is not None else None,
        "object_score": float(unit["object_score"].float().mean())
        if unit.get("object_score") is not None else None,
    }


def corrupt_mask(mask, level, kind, frame_index):
    """Deterministic E3 corruption; level is fixed before the run."""
    mask = np.asarray(mask, dtype=np.uint8)
    if not 0 <= level <= 1:
        raise ValueError("--corruption-level must be in [0, 1]")
    if level == 0:
        return mask.astype(bool)
    if kind == "mixed":
        kind = ("erosion", "dilation", "shift", "blob", "deletion")[frame_index % 5]
    radius = max(1, round(1 + 10 * level))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    if kind == "erosion":
        result = cv2.erode(mask, kernel)
    elif kind == "dilation":
        result = cv2.dilate(mask, kernel)
    elif kind == "shift":
        dy, dx = round(mask.shape[0] * 0.2 * level), round(mask.shape[1] * 0.2 * level)
        result = np.zeros_like(mask)
        result[dy:, dx:] = mask[: mask.shape[0] - dy, : mask.shape[1] - dx]
    elif kind == "blob":
        result = mask.copy()
        centre = (int(mask.shape[1] * 0.2), int(mask.shape[0] * 0.2))
        axes = (max(1, round(mask.shape[1] * 0.12 * level)),
                max(1, round(mask.shape[0] * 0.12 * level)))
        cv2.ellipse(result, centre, axes, 0, 0, 360, 1, -1)
    elif kind == "deletion":
        result = mask.copy()
        cutoff = round(mask.shape[1] * (1 - 0.8 * level))
        result[:, cutoff:] = 0
    else:
        raise ValueError(kind)
    return result.astype(bool)


def replace_spatial_memory(predictor, state, output_dict, frames, frame_index, label_id,
                           all_label_ids, source, corruption_level, corruption_type):
    if source == "predicted":
        return
    target = load_label_mask(frames[frame_index].mask_path, all_label_ids) == label_id
    if source == "corrupted":
        target = corrupt_mask(target, corruption_level, corruption_type, frame_index)
    logits = torch.from_numpy(target).to(predictor.device).float()[None, None] * 20 - 10
    score = torch.full((1, 1), 10.0 if target.any() else -10.0, device=predictor.device)
    features, position = predictor._run_memory_encoder(
        inference_state=state, frame_idx=frame_index, batch_size=1,
        high_res_masks=logits, object_score_logits=score, is_mask_from_pts=True,
    )
    storage = "cond_frame_outputs" if frame_index in output_dict["cond_frame_outputs"] else "non_cond_frame_outputs"
    output_dict[storage][frame_index]["maskmem_features"] = features
    output_dict[storage][frame_index]["maskmem_pos_enc"] = position


@torch.inference_mode()
def analyze_video(predictor, controller, video_id, frames, label_id, all_label_ids, protocol, rng,
                  frame_budget, frame_selection, every_n, random_null, memory_source,
                  corruption_level, corruption_type):
    records, rows = [], []
    with tempfile.TemporaryDirectory(prefix="utility_frames_") as temporary:
        staged = Path(temporary)
        stage_frames(frames, staged)
        baseline_logits, prompt, _, _ = predict_label(
            predictor, frames, label_id, all_label_ids, False, staged, False, "mask"
        )
        if prompt is None:
            return records, rows
        baseline_dice = [
            dice(baseline_logits[index] > 0, load_label_mask(frame.mask_path, all_label_ids) == label_id)
            for index, frame in enumerate(frames)
        ]
        if frame_selection == "failure_onsets":
            episode_starts = [
                prompt + 1 + episode.start
                for episode in failure_episodes(
                    baseline_dice[prompt + 1 :],
                    protocol["failure_definition"]["threshold"],
                    protocol["failure_definition"]["minimum_consecutive_frames"],
                )
            ]
        elif frame_selection == "all":
            episode_starts = list(range(prompt + 1, len(frames)))
        else:
            if every_n < 1:
                raise ValueError("--every-n must be positive")
            episode_starts = list(range(prompt + 1, len(frames), every_n))
        if frame_budget is not None:
            episode_starts = episode_starts[:frame_budget]

        state = predictor.init_state(
            video_path=str(staged), offload_video_to_cpu=True,
            offload_state_to_cpu=predictor.device.type == "cuda",
        )
        try:
            prompt_mask = load_label_mask(frames[prompt].mask_path, all_label_ids) == label_id
            predictor.add_new_mask(
                state, frame_idx=prompt, obj_id=1,
                mask=torch.from_numpy(prompt_mask),
            )
            output_dict = state["output_dict"]
            for frame_index, _, mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=prompt
            ):
                if frame_index not in episode_starts:
                    replace_spatial_memory(
                        predictor, state, output_dict, frames, frame_index, label_id,
                        all_label_ids, memory_source, corruption_level, corruption_type,
                    )
                    continue
                target = load_label_mask(frames[frame_index].mask_path, all_label_ids) == label_id
                baseline_prediction = mask_logits[0, 0].detach().cpu().numpy() > 0
                baseline_score = dice(baseline_prediction, target)
                trace = predictor.memory_attention.last_trace
                units = trace["units"]
                if not units:
                    replace_spatial_memory(
                        predictor, state, output_dict, frames, frame_index, label_id,
                        all_label_ids, memory_source, corruption_level, corruption_type,
                    )
                    continue
                features = trace["utility_features"][0].detach().cpu()
                attention = _attention_by_unit(trace)
                reconstruction_error = max(
                    [value for value in trace["reconstruction_max_abs"] if value is not None],
                    default=float("nan"),
                )

                def score_excluding(frame_indices):
                    controller.excluded_frame_indices = set(frame_indices)
                    _, rerun_logits = predictor.run_single_frame_inference(
                        inference_state=state, output_dict=output_dict,
                        frame_idx=frame_index, batch_size=1,
                        is_init_cond_frame=False, point_inputs=None, mask_inputs=None,
                        reverse=False, run_mem_encoder=False,
                    )
                    return dice(_mask_at_original_size(rerun_logits, target.shape), target), rerun_logits

                no_op_score, no_op_logits = score_excluding(set())
                stored = output_dict["non_cond_frame_outputs"][frame_index]["pred_masks"]
                no_op_max_abs = float((no_op_logits.float().cpu() - stored.float().cpu()).abs().max())
                utilities = []
                for unit in units:
                    excluded_score, _ = score_excluding({unit["frame_idx"]})
                    utilities.append(baseline_score - excluded_score)

                harmful = {unit["frame_idx"] for unit, utility in zip(units, utilities) if utility < 0}
                oracle_score, _ = score_excluding(harmful) if harmful else (baseline_score, None)
                random_scores = []
                if harmful and random_null:
                    candidates = [unit["frame_idx"] for unit in units]
                    for _ in range(len(units)):
                        selected = rng.choice(candidates, size=len(harmful), replace=False)
                        random_scores.append(score_excluding(set(map(int, selected)))[0])
                controller.excluded_frame_indices = set()
                best_random = max(random_scores, default=baseline_score)

                target_rows, target_columns = np.nonzero(target)
                if target_rows.size:
                    target_box = np.array([
                        target_columns.min(), target_rows.min(),
                        target_columns.max(), target_rows.max(),
                    ], dtype=np.float32)
                    point_inputs = predictor.prepare_point_inputs(
                        inference_state=state, box=target_box, points=None, labels=None,
                        normalize_coords=True,
                    )
                    _, oracle_prompt_logits = predictor.run_single_frame_inference(
                        inference_state=state, output_dict=output_dict,
                        frame_idx=frame_index, batch_size=1,
                        is_init_cond_frame=False, point_inputs=point_inputs,
                        mask_inputs=None, reverse=False, run_mem_encoder=False,
                        prev_sam_mask_logits=stored.to(predictor.device),
                    )
                    oracle_prompt_score = dice(
                        _mask_at_original_size(oracle_prompt_logits, target.shape), target
                    )
                else:
                    oracle_prompt_score = float("nan")

                agreement = rank_agreement(attention, utilities)
                rows.append({
                    "video": video_id, "frame": frame_index, "baseline_dice": baseline_score,
                    "memory_source": memory_source, "corruption_level": corruption_level,
                    "corruption_type": corruption_type,
                    "no_op_dice": no_op_score, "no_op_max_abs": no_op_max_abs,
                    "oracle_retrieval_dice": oracle_score,
                    "best_of_n_random_dice": best_random, "random_n": len(units),
                    "oracle_prompt_dice": oracle_prompt_score,
                    "excluded_harmful_units": len(harmful),
                    "attention_spearman": agreement["spearman"],
                    "attention_kendall": agreement["kendall"],
                    "attention_reconstruction_max_abs": reconstruction_error,
                    "units": [_json_unit(unit) for unit in units],
                    "utility": utilities, "attention": attention,
                })
                query_dim, memory_dim = predictor.hidden_dim, predictor.mem_dim
                anchor_index = next(
                    (index for index, unit in enumerate(units) if unit["conditioning"]), 0
                )
                anchor_key = features[
                    anchor_index, 2 * query_dim : 2 * query_dim + memory_dim
                ]
                for index, (unit, utility) in enumerate(zip(units, utilities)):
                    feature = features[index]
                    memory_key = feature[2 * query_dim : 2 * query_dim + memory_dim]
                    similarity = F.cosine_similarity(
                        anchor_key, memory_key, dim=0
                    ).item()
                    unit_json = _json_unit(unit)
                    confidence = unit_json["predicted_iou"]
                    if confidence is not None and unit_json["object_score"] is not None:
                        confidence *= float(torch.sigmoid(torch.tensor(unit_json["object_score"])))
                    records.append({
                        "video": video_id, "frame": frame_index,
                        "memory_frame": unit["frame_idx"],
                        "features": feature, "utility": float(utility),
                        "attention": float(attention[index]),
                        "semantic_similarity": float(similarity),
                        "confidence": confidence,
                        "object_score": unit_json["object_score"],
                    })
                replace_spatial_memory(
                    predictor, state, output_dict, frames, frame_index, label_id,
                    all_label_ids, memory_source, corruption_level, corruption_type,
                )
        finally:
            controller.excluded_frame_indices = set()
            predictor.reset_state(state)
    return records, rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(f"[utility] stage starting: output={args.output_dir}", flush=True)
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    videos = load_manifest(args.manifest, args.split)
    if args.video_ids:
        videos = {name: frames for name, frames in videos.items() if name in set(args.video_ids)}
    if not videos:
        raise ValueError("No requested videos are present in the selected split.")

    setup = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "videos": sorted(videos),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.sam2_checkpoint.read_bytes()).hexdigest(),
        "utility_definition": "baseline Dice - Dice when one spatial-memory frame is excluded",
        "object_pointer_tokens_modulated": False,
    }
    (args.output_dir / "setup.json").write_text(json.dumps(setup, indent=2) + "\n")
    run = start_wandb(
        args.wandb_project, args.wandb_entity, args.wandb_name,
        args.wandb_mode, args.wandb_base_url, setup, args.output_dir,
        ("utility", "E1", "E2", "E4"),
    )
    all_records, all_rows = [], []
    save_progress(args.output_dir, all_records, all_rows, state="building_model")
    print("[utility] W&B initialized; building predictor", flush=True)
    predictor = build_video_predictor(
        args.sam2_cfg, args.sam2_checkpoint, args.device
    )
    controller = UtilityController(
        UtilityHead(query_dim=predictor.hidden_dim, memory_dim=predictor.mem_dim)
    )
    controller.enabled = False
    attach_utility_controller(predictor, controller, capture_attention=True)
    save_progress(args.output_dir, all_records, all_rows, state="running")
    print(f"[utility] predictor ready; videos={len(videos)}", flush=True)
    remaining = args.max_analyzed_frames
    try:
        for video_id, frames in videos.items():
            if remaining is not None and remaining <= 0:
                break
            available = sorted(int(value) for value in np.unique(load_semantic_mask(frames[0].mask_path)) if value)
            label_ids = args.label_ids or available
            missing = sorted(set(label_ids) - set(available))
            if missing:
                raise ValueError(f"{video_id}: requested object IDs absent from first mask: {missing}")
            for label_id in label_ids:
                records, rows = analyze_video(
                    predictor, controller, f"{video_id}:{label_id}", frames, label_id,
                    available, protocol, rng, remaining, args.frame_selection, args.every_n,
                    args.random_null,
                    args.memory_source, args.corruption_level, args.corruption_type,
                )
                all_records.extend(records)
                all_rows.extend(rows)
                save_progress(
                    args.output_dir, all_records, all_rows,
                    state="running", video=video_id, label_id=label_id,
                )
                if remaining is not None:
                    remaining -= len(rows)
                for row in rows:
                    log(run, "E2", {
                        "baseline_dice": row["baseline_dice"],
                        "oracle_retrieval_gain": row["oracle_retrieval_dice"] - row["baseline_dice"],
                        "best_of_n_random_gain": row["best_of_n_random_dice"] - row["baseline_dice"],
                        "attention_utility_spearman": row["attention_spearman"],
                    }, len(all_rows))
                print(f"{video_id}:{label_id}: {len(rows)} frames analyzed ({args.frame_selection})", flush=True)
                if remaining is not None and remaining <= 0:
                    break
    finally:
        finish(run)

    save_progress(args.output_dir, all_records, all_rows, state="complete")
    summary = {
        "analyzed_frames": len(all_rows), "utility_units": len(all_records),
        "mean_baseline_dice": float(np.mean([row["baseline_dice"] for row in all_rows])) if all_rows else None,
        "mean_oracle_retrieval_gain": float(np.mean([
            row["oracle_retrieval_dice"] - row["baseline_dice"] for row in all_rows
        ])) if all_rows else None,
        "mean_best_of_n_random_gain": float(np.mean([
            row["best_of_n_random_dice"] - row["baseline_dice"] for row in all_rows
        ])) if all_rows else None,
        "sanity_no_op_max_abs": max([row["no_op_max_abs"] for row in all_rows], default=None),
        "sanity_attention_reconstruction_max_abs": max([
            row["attention_reconstruction_max_abs"] for row in all_rows
            if np.isfinite(row["attention_reconstruction_max_abs"])
        ], default=None),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
