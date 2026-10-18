#!/usr/bin/env python3
"""Bounded v2 diagnostics, reusing native SAM2 propagation and study helpers.

No new segmentation architecture or training loop. Audit uses labels only for
sampling feasibility. Diagnosis uses one initial mask and evaluates labels only
after prediction. Spatial suppression is explicitly not full memory-off.
"""

# ruff: noqa: E402
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "external/MedSAM2")]
from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from evaluation.statistics import temporal_summary
from modeling.medsam2 import build_video_predictor
from modeling.region_tokens import (
    CATEGORIES,
    capture_chunked_region_attention,
    disjoint_distance_match,
    indices_to_masks,
    matched_experiments,
    persistent_region_rollout,
    region_geometry,
    regional_bias_candidates,
    region_memory_mask,
    subsampled_disjoint_match,
    token_categories,
)
from modeling.memory_attention import (
    UtilityController,
    attach_utility_controller,
)
from scripts.audit_region_initialization import prompt_index, target_indices
from modeling.memory_interventions import replace_spatial_memory
from scripts.run_region_experiments import digest, validate_output
from scripts.run_region_token_pilot import metrics, query_dice_harm, write_json
from training.tracking import log_results_artifact, start_wandb


def synchronize_storage(device):
    """Finish async accelerator-to-CPU offloads before Python deep-copies banks."""
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def audit(videos, output, run, side_matched=False):
    totals = {}
    with (output / "feasibility.jsonl").open("w") as stream:
        for video, frames in videos.items():
            counts = {
                "frames": 0,
                "boundary_feasible": 0,
                "mixed_feasible": 0,
                "boundary_tokens": 0,
                "mixed_tokens": 0,
                "empty_masks": 0,
            }
            for frame in frames:
                mask = load_label_mask(frame.mask_path, [1]) == 1
                geometry = region_geometry(mask)
                strata = geometry["distance_bin"]
                if side_matched:
                    strata = 2 * strata + geometry["inside"].astype(np.int8)
                boundary = token_categories(mask)["categories"] == 0
                row = {
                    "video": video,
                    "frame": frame.frame_index,
                    "oracle_geometry": True,
                }
                counts["frames"] += 1
                counts["empty_masks"] += int(not mask.any())
                for name, target in (
                    ("boundary", boundary),
                    ("mixed", geometry["mixed"]),
                ):
                    selected = disjoint_distance_match(
                        target, ~target, strata, seed=20261018
                    )
                    size = int(target.sum())
                    row[name] = {
                        "target_tokens": size,
                        "feasible": selected is not None,
                        "control_overlap": 0 if selected is not None else None,
                    }
                    counts[f"{name}_tokens"] += size
                    counts[f"{name}_feasible"] += int(selected is not None)
                stream.write(json.dumps(row) + "\n")
            totals[video] = counts
            run.log(
                {"audit/video": video, **{f"audit/{k}": v for k, v in counts.items()}}
            )
            write_json(
                output / "summary.json",
                {
                    "phase": "feasibility",
                    "completed": False,
                    "sequences": totals,
                    "performance_measured": False,
                },
            )
            print(f"[v2 audit] {video} {counts}", flush=True)
    return {
        "phase": "feasibility",
        "completed": True,
        "sequences": totals,
        "performance_measured": False,
    }


@torch.inference_mode()
def counterfactual(videos, args, output, run):
    """Sampled A1/A3 perturbations; independent cached-bank replays, no writeback.

    Native image/memory encoder, masking, metrics and samplers are reused. Full
    attention matrices are not materialized on the 24-GB Mac. This does not
    replace the unchanged Kaggle E2 or establish accumulated-corruption effects.
    """
    expected = json.loads(
        (ROOT / "protocols/region_tokens_v2_amendment.json").read_text()
    )["model"]["checkpoint_sha256"]
    if digest(args.checkpoint) != expected:
        raise ValueError("Unregistered checkpoint")
    predictor = build_video_predictor(
        "configs/sam2.1_hiera_b+1024.yaml", args.checkpoint, args.device
    ).eval()
    controller = UtilityController(learn_scale=False)
    controller.enabled = False
    attach_utility_controller(predictor, controller)
    totals, count = {}, 0
    with (output / "frames.jsonl").open("w") as stream:
        for video in args.sequence:
            frames = videos[video]
            prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
            targets = set(target_indices(len(frames), prompt, 2))
            totals[video] = {"targets": sorted(targets), "completed_conditions": 0}
            with tempfile.TemporaryDirectory(
                prefix="replay_frames_", dir=output
            ) as directory:
                stage_frames(frames, Path(directory))
                state = predictor.init_state(
                    video_path=directory,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )
                try:
                    initial = load_label_mask(frames[prompt].mask_path, [1]) == 1
                    predictor.add_new_mask(
                        state,
                        frame_idx=prompt,
                        obj_id=1,
                        mask=torch.from_numpy(initial),
                    )
                    for t, _, native_logits in predictor.propagate_in_video(
                        state, start_frame_idx=prompt
                    ):
                        if (
                            args.deadline
                            and datetime.now().astimezone()
                            >= datetime.fromisoformat(args.deadline)
                        ):
                            return {
                                "completed": False,
                                "reason": "deadline",
                                "sequences": totals,
                                "rows": count,
                            }
                        if t not in targets:
                            continue
                        synchronize_storage(args.device)
                        original = copy.deepcopy(state["output_dict"])
                        original["non_cond_frame_outputs"].pop(t, None)
                        cached = copy.deepcopy(state["cached_features"])
                        trace = predictor.memory_attention.last_trace
                        units, queries = trace["units"], trace["query_regions"]
                        if len(units) != 7:
                            raise RuntimeError("Unexpected default memory bank")
                        truth = load_label_mask(frames[t].mask_path, [1]) == 1
                        reference_masks = {
                            u["frame_idx"]: load_label_mask(
                                frames[u["frame_idx"]].mask_path, [1]
                            )
                            == 1
                            for u in units
                        }
                        taxonomy = {
                            index: token_categories(mask)["categories"]
                            for index, mask in reference_masks.items()
                        }
                        for condition in (
                            ("natural", "c0")
                            if args.regional_followup
                            else ("natural", "c0", "c1", "c2")
                        ):
                            bank = copy.deepcopy(original)
                            if condition != "natural":
                                for unit in units:
                                    index = unit["frame_idx"]
                                    replacement = region_memory_mask(
                                        reference_masks[index],
                                        condition,
                                        index,
                                        bool(unit["conditioning"]),
                                    )
                                    replace_spatial_memory(
                                        predictor,
                                        state,
                                        bank,
                                        frames,
                                        index,
                                        1,
                                        [1],
                                        "ground_truth",
                                        0,
                                        "mixed",
                                        mask_override=replacement,
                                    )
                            # The upstream encoder offloads maskmem_features
                            # with non_blocking=True. A CPU deepcopy before the
                            # device fence can observe unfinished host buffers.
                            synchronize_storage(args.device)

                            def replay(masks=None, biases=None):
                                state["cached_features"] = copy.deepcopy(cached)
                                controller.excluded_token_masks = masks or {}
                                controller.token_biases = biases or {}
                                return (
                                    predictor.run_single_frame_inference(
                                        inference_state=state,
                                        output_dict=copy.deepcopy(bank),
                                        frame_idx=t,
                                        batch_size=1,
                                        is_init_cond_frame=False,
                                        point_inputs=None,
                                        mask_inputs=None,
                                        reverse=False,
                                        run_mem_encoder=False,
                                    )[1]
                                    .detach()
                                    .clone()
                                )

                            baseline = replay()
                            if args.attention:
                                with capture_chunked_region_attention(
                                    predictor.memory_attention,
                                    units,
                                    taxonomy,
                                    failure_dir=output
                                    / "attention_failure_tensors"
                                    / f"{video}_{t}_{condition}",
                                ) as attention:
                                    observed = replay()
                                if not torch.equal(baseline, observed):
                                    raise RuntimeError(
                                        "Attention measurement changed native logits"
                                    )
                                if len(attention) != len(
                                    predictor.memory_attention.layers
                                ) or any(
                                    not np.isfinite(a["reconstruction_max_abs"])
                                    or a["reconstruction_max_abs"] > 1e-4
                                    or abs(a["total_mass"] - 1) > 1e-4
                                    for a in attention
                                ):
                                    write_json(
                                        output / "attention_failure.json",
                                        {
                                            "video": video,
                                            "frame": t,
                                            "condition": condition,
                                            "expected_layers": len(
                                                predictor.memory_attention.layers
                                            ),
                                            "logits_bit_identical": True,
                                            "reconstruction_absolute_limit": 1e-4,
                                            "mass_absolute_limit": 1e-4,
                                            "layers": attention,
                                        },
                                    )
                                    raise RuntimeError(
                                        "Chunked attention reconstruction or mass check failed"
                                    )
                                with (output / "attention.jsonl").open(
                                    "a"
                                ) as attention_stream:
                                    attention_stream.write(
                                        json.dumps(
                                            {
                                                "video": video,
                                                "frame": t,
                                                "condition": condition,
                                                "region_source": "fixed oracle GT supports; analysis only",
                                                "logits_bit_identical": True,
                                                "layers": attention,
                                            },
                                            allow_nan=False,
                                        )
                                        + "\n"
                                    )
                            restored = replay()
                            if not torch.equal(baseline, restored):
                                write_json(
                                    output / "restore_failure.json",
                                    {
                                        "video": video,
                                        "frame": t,
                                        "condition": condition,
                                        "baseline_finite": bool(
                                            torch.isfinite(baseline).all()
                                        ),
                                        "restored_finite": bool(
                                            torch.isfinite(restored).all()
                                        ),
                                        "max_abs": float(
                                            torch.nan_to_num(
                                                (baseline - restored).abs()
                                            ).max()
                                        ),
                                        "shape": list(baseline.shape),
                                    },
                                )
                                torch.save(
                                    {
                                        "baseline": baseline.cpu(),
                                        "restored": restored.cpu(),
                                    },
                                    output / "restore_failure.pt",
                                )
                                raise RuntimeError(
                                    "Independent cached-bank restore failed"
                                )
                            if condition == "natural":
                                # Public single-frame inference returns decoder-
                                # resolution logits; propagation returns video-
                                # resolution logits. Compare via the very same
                                # native formatter, not mismatched shapes.
                                _, replay_video = predictor._get_orig_video_res_output(
                                    state, baseline
                                )
                                if not torch.equal(replay_video, native_logits):
                                    raise RuntimeError(
                                        "Natural replay differs from native propagation at video resolution"
                                    )
                            base = metrics(baseline, truth)
                            if args.regional_followup:
                                predicted_categories = {}
                                for u in units:
                                    if not u["conditioning"]:
                                        index = u["frame_idx"]
                                        stored = original["non_cond_frame_outputs"][
                                            index
                                        ]
                                        predicted = (
                                            stored["pred_masks"][0, 0]
                                            .detach()
                                            .cpu()
                                            .numpy()
                                            > 0
                                        )
                                        predicted_categories[index] = token_categories(
                                            predicted
                                        )["categories"]
                                # Actual zero-bias dictionary must reproduce native logits,
                                # not merely an empty/no-op mask control.
                                zero = {
                                    i: np.zeros(x.size, np.float32)
                                    for i, x in predicted_categories.items()
                                }
                                if not torch.equal(baseline, replay(biases=zero)):
                                    raise RuntimeError(
                                        "Explicit zero regional bias changed logits"
                                    )
                                for search_seed in (None, 0, 1, 2):
                                    candidates = []
                                    for candidate, biases in regional_bias_candidates(
                                        predicted_categories, search_seed
                                    ):
                                        altered = replay(biases=biases)
                                        candidates.append(
                                            {
                                                "candidate": candidate,
                                                "metrics": metrics(altered, truth),
                                                "token_l2": float(
                                                    np.sqrt(
                                                        sum(
                                                            np.square(
                                                                x.astype(np.float64)
                                                            ).sum()
                                                            for x in biases.values()
                                                        )
                                                    )
                                                ),
                                            }
                                        )
                                    with (output / "headroom.jsonl").open(
                                        "a"
                                    ) as headroom_stream:
                                        headroom_stream.write(
                                            json.dumps(
                                                {
                                                    "video": video,
                                                    "frame": t,
                                                    "condition": condition,
                                                    "search": "regional"
                                                    if search_seed is None
                                                    else "random",
                                                    "seed": search_seed,
                                                    "baseline": base,
                                                    "candidates": candidates,
                                                    "region_source": "stored native predicted masks; fixed before interventions",
                                                    "selection": "oracle target Dice; no writeback; not deployable",
                                                    "explicit_zero_bias_bit_identical": True,
                                                },
                                                allow_nan=False,
                                            )
                                            + "\n"
                                        )
                                    print(
                                        f"[headroom] {video} target={t} condition={condition} seed={search_seed}",
                                        flush=True,
                                    )
                                controller.token_biases = {}
                            for bank_name, conditioning in (
                                (("recent", False),)
                                if args.regional_followup
                                else (("conditioning", True), ("recent", False))
                            ):
                                chosen = [
                                    u
                                    for u in units
                                    if bool(u["conditioning"]) == conditioning
                                ]
                                labels = np.stack(
                                    [taxonomy[u["frame_idx"]] for u in chosen]
                                )
                                k, reference, experiments = matched_experiments(
                                    labels, spatial_match="frame_distance"
                                )
                                # Prospective disjoint mixed-coverage comparison, same fixed GT supports.
                                geometries = [
                                    region_geometry(reference_masks[u["frame_idx"]])
                                    for u in chosen
                                ]
                                mixed = np.stack([g["mixed"] for g in geometries])
                                bins = np.stack(
                                    [
                                        2 * g["distance_bin"]
                                        + g["inside"].astype(np.int8)
                                        for g in geometries
                                    ]
                                )
                                mixed_control = disjoint_distance_match(
                                    mixed, ~mixed, bins, seed=0
                                )
                                extra = [
                                    ("mixed", 0, np.flatnonzero(mixed)),
                                    ("mixed_disjoint_control", 0, mixed_control),
                                ]
                                support_metadata = {}
                                if args.regional_followup:
                                    exact = np.stack(
                                        [
                                            np.where(
                                                np.isfinite(g["absolute_distance"]),
                                                np.rint(
                                                    np.nan_to_num(
                                                        g["absolute_distance"]
                                                    )
                                                    ** 2
                                                ),
                                                -1,
                                            ).astype(np.int32)
                                            for g in geometries
                                        ]
                                    )
                                    experiments, extra = [], []
                                    for category, name in enumerate(
                                        (*CATEGORIES, "mixed_side")
                                    ):
                                        support = (
                                            labels == category
                                            if category < 4
                                            else mixed
                                        )
                                        strata = (
                                            exact
                                            if category < 4
                                            else np.where(
                                                exact >= 0,
                                                2 * exact
                                                + np.stack(
                                                    [g["inside"] for g in geometries]
                                                ),
                                                -1,
                                            )
                                        )
                                        for seed in (0, 1, 2):
                                            left, right, info = (
                                                subsampled_disjoint_match(
                                                    support,
                                                    ~support,
                                                    strata,
                                                    cap=64,
                                                    seed=seed,
                                                )
                                            )
                                            for role, selected in (
                                                ("target", left),
                                                ("control", right),
                                            ):
                                                key = f"paired_{name}_{role}"
                                                experiments.append(
                                                    (key, seed, selected)
                                                )
                                                support_metadata[(key, seed)] = {
                                                    **info,
                                                    "region": name,
                                                    "role": role,
                                                    "strata": "frame/exact squared distance"
                                                    + (
                                                        "/side" if category == 4 else ""
                                                    ),
                                                }
                                for category, seed, selected in experiments + extra:
                                    budget = (
                                        support_metadata[(category, seed)]["selected"]
                                        if args.regional_followup
                                        else int(mixed.sum())
                                        if category.startswith("mixed")
                                        else k
                                    )
                                    row = {
                                        "video": video,
                                        "frame": t,
                                        "condition": condition,
                                        "bank": bank_name,
                                        "category": category,
                                        "seed": seed,
                                        "k": budget,
                                        "baseline": base,
                                        "category_source": "fixed oracle GT supports across conditions",
                                        "condition_scope": "target-bank perturbation; natural pointers",
                                        "matched_random": "frame_distance, overlap allowed",
                                        "mixed_control_match": "frame, absolute-distance bin and inside/outside; disjoint",
                                    }
                                    if args.regional_followup:
                                        row.update(
                                            experiment="subsampled_disjoint_geometry",
                                            matching=support_metadata[(category, seed)],
                                            matched_random="disjoint per-region control, exact frame/distance strata",
                                            mixed_control_match="exact frame/distance/side only for mixed_side",
                                        )
                                    if selected is None or budget == 0:
                                        row.update(
                                            state="skipped",
                                            reason="no_common_support"
                                            if args.regional_followup
                                            else "infeasible_full_budget",
                                        )
                                    else:
                                        altered = replay(
                                            indices_to_masks(chosen, selected)
                                        )
                                        values = metrics(altered, truth)
                                        row.update(
                                            state="complete",
                                            intervention=values,
                                            harm={
                                                name: base[name] - values[name]
                                                if base[name] is not None
                                                and values[name] is not None
                                                else None
                                                for name in (
                                                    "dice",
                                                    "iou",
                                                    "boundary_f",
                                                )
                                            },
                                            query_dice_harm=query_dice_harm(
                                                baseline, altered, truth, queries
                                            ),
                                            reference_overlap=float(
                                                np.isin(selected, reference).mean()
                                            ),
                                        )
                                    stream.write(
                                        json.dumps(row, allow_nan=False) + "\n"
                                    )
                                    stream.flush()
                                    count += 1
                            controller.excluded_token_masks = {}
                            totals[video]["completed_conditions"] += 1
                            run.log(
                                {
                                    "video": video,
                                    "frame": t,
                                    "condition": condition,
                                    "rows": count,
                                    "baseline/dice": base["dice"],
                                }
                            )
                            write_json(
                                output / "summary.json",
                                {
                                    "completed": False,
                                    "sequences": totals,
                                    "rows": count,
                                },
                            )
                            print(
                                f"[v2 replay] {video} target={t} condition={condition} rows={count}",
                                flush=True,
                            )
                        state["cached_features"] = cached
                        controller.excluded_token_masks = {}
                        controller.token_biases = {}
                    log_results_artifact(
                        run,
                        output,
                        ["summary.json", "frames.jsonl"]
                        + (["attention.jsonl"] if args.attention else []),
                        # headroom is saved alongside the intervention rows.
                    )
                    if args.regional_followup:
                        log_results_artifact(
                            run,
                            output,
                            ["headroom.jsonl", "frames.jsonl", "summary.json"],
                        )
                finally:
                    controller.excluded_token_masks = {}
                    controller.token_biases = {}
                    predictor.reset_state(state)
            if args.device == "mps":
                torch.mps.empty_cache()
    return {
        "phase": "counterfactual",
        "completed": True,
        "sequences": totals,
        "rows": count,
        "full_A3_complete": False,
        "scope": "sampled target-bank perturbation, not accumulated corrupted rollouts",
    }


@torch.inference_mode()
def diagnose(videos, args, output, run):
    expected = json.loads(
        (ROOT / "protocols/region_tokens_v2_amendment.json").read_text()
    )["model"]["checkpoint_sha256"]
    if digest(args.checkpoint) != expected:
        raise ValueError("B+ checkpoint hash does not match registered model")
    predictor = build_video_predictor(
        "configs/sam2.1_hiera_b+1024.yaml", args.checkpoint, args.device
    ).eval()
    controller = UtilityController(learn_scale=False)
    controller.enabled = False
    attach_utility_controller(predictor, controller)
    reads = [0]
    handle = predictor.memory_attention.register_forward_hook(
        lambda _m, _i, _o: reads.__setitem__(0, reads[0] + 1)
    )
    arms = (
        ["native"]
        if args.phase == "baseline"
        else ["native", "zero_controller"]
        if args.phase == "validate"
        else ["boundary", "foreground", "near_background", "far_background"]
        + [f"random_{seed}" for seed in range(5)]
        if args.phase == "persistent"
        else ["native", "spatial_suppressed"]
    )
    results, signatures = {}, {}
    try:
        for video in args.sequence:
            original_frames = videos[video]
            prompt = prompt_index(
                original_frames, {"initialization": "first_labeled_frame"}
            )
            frames = (
                original_frames[: prompt + args.max_frames]
                if args.max_frames
                else original_frames
            )
            for arm in arms:
                if (
                    args.deadline
                    and datetime.now().astimezone()
                    >= datetime.fromisoformat(args.deadline)
                ):
                    return {
                        "completed": False,
                        "reason": "deadline",
                        "sequences": results,
                    }
                predictor.memory_attention.utility_controller = (
                    None if arm == "native" else controller
                )
                controller.token_mask_policy = (
                    (
                        lambda units: {
                            u["frame_idx"]: torch.ones(
                                u["end"] - u["start"], dtype=torch.bool
                            )
                            for u in units
                        }
                    )
                    if arm == "spatial_suppressed"
                    else None
                )
                controller.excluded_token_masks = {}
                folder = output / video / arm
                folder.mkdir(parents=True, exist_ok=False)
                scores, hashes, started, before_reads = (
                    [],
                    [],
                    time.perf_counter(),
                    reads[0],
                )
                policy_diagnostics = []
                complete = True
                with tempfile.TemporaryDirectory(
                    prefix="frames_", dir=output
                ) as directory:
                    stage_frames(frames, Path(directory))
                    state = predictor.init_state(
                        video_path=directory,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                    )
                    try:
                        initial = load_label_mask(frames[prompt].mask_path, [1]) == 1
                        predictor.add_new_mask(
                            state,
                            frame_idx=prompt,
                            obj_id=1,
                            mask=torch.from_numpy(initial),
                        )
                        predicted_masks = {prompt: initial}
                        propagation = (
                            persistent_region_rollout(
                                predictor,
                                state,
                                controller,
                                lambda index: predicted_masks[index],
                                "random" if arm.startswith("random_") else arm,
                                seed=int(arm.split("_")[1])
                                if arm.startswith("random_")
                                else 0,
                                diagnostics=policy_diagnostics,
                            )
                            if args.phase == "persistent"
                            else predictor.propagate_in_video(
                                state, start_frame_idx=prompt
                            )
                        )
                        with (folder / "frames.jsonl").open("w") as stream:
                            for t, _, logits in propagation:
                                array = logits[0, 0].float().cpu().numpy()
                                predicted_masks[t] = array > 0
                                hashes.append(
                                    hashlib.sha256(array.tobytes()).hexdigest()
                                )
                                Image.fromarray(
                                    (array > 0).astype(np.uint8) * 255
                                ).save(folder / f"{t:06d}.png")
                                if t > prompt:
                                    target = (
                                        load_label_mask(frames[t].mask_path, [1]) == 1
                                    )
                                    values = metrics(
                                        logits[:, :1], target, decoder_logits=False
                                    )
                                    scores.append(values["dice"])
                                    row = {
                                        "video": video,
                                        "arm": arm,
                                        "frame": t,
                                        "metrics": values,
                                        "elapsed_seconds": time.perf_counter()
                                        - started,
                                        "memory_reads": reads[0] - before_reads,
                                        "policy": policy_diagnostics[-1]
                                        if policy_diagnostics
                                        else None,
                                    }
                                    stream.write(
                                        json.dumps(row, allow_nan=False) + "\n"
                                    )
                                    stream.flush()
                                    run.log(
                                        {
                                            "video": video,
                                            "arm": arm,
                                            "frame": t,
                                            **{
                                                f"metrics/{k}": v
                                                for k, v in values.items()
                                                if v is not None
                                            },
                                        }
                                    )
                                    write_json(output / "progress.json", row)
                                    print(
                                        f"[v2] {video} {arm} frame={t} dice={values['dice']:.4f}",
                                        flush=True,
                                    )
                                if (
                                    args.deadline
                                    and datetime.now().astimezone()
                                    >= datetime.fromisoformat(args.deadline)
                                ):
                                    complete = False
                                    break
                    finally:
                        predictor.reset_state(state)
                result = {
                    "completed_selected_interval": complete,
                    "full_sequence": complete and len(frames) == len(original_frames),
                    "evaluated_frames": len(scores),
                    "mean_dice": float(np.mean(scores)) if scores else None,
                    "temporal": temporal_summary(scores),
                    "seconds": time.perf_counter() - started,
                    "memory_reads": reads[0] - before_reads,
                    "prompt": prompt,
                    "pointer_tokens_retained": True,
                    "policy_reads": len(policy_diagnostics),
                    "infeasible_policy_reads": sum(
                        not d["feasible"] for d in policy_diagnostics
                    ),
                }
                results.setdefault(video, {})[arm] = result
                signatures[(video, arm)] = hashes
                if args.phase == "validate" and arm == "zero_controller":
                    result["logits_bit_identical"] = (
                        hashes == signatures[(video, "native")]
                    )
                    if not result["logits_bit_identical"]:
                        raise RuntimeError("MPS no-op validation failed")
                write_json(
                    output / "summary.json", {"completed": False, "sequences": results}
                )
                log_results_artifact(
                    run,
                    output,
                    ["summary.json", "progress.json", f"{video}/{arm}/frames.jsonl"],
                )
                if args.device == "mps":
                    torch.mps.empty_cache()
                if not complete:
                    return {
                        "completed": False,
                        "reason": "deadline",
                        "sequences": results,
                    }
    finally:
        handle.remove()
    return {
        "completed": True,
        "phase": args.phase,
        "sequences": results,
        "population_inference": False,
        "spatial_suppression_is_full_memory_off": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=[
            "audit",
            "validate",
            "diagnose",
            "counterfactual",
            "persistent",
            "baseline",
        ],
        required=True,
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl"
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--deadline")
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--diagnosis-dir", type=Path)
    parser.add_argument("--side-matched", action="store_true")
    parser.add_argument(
        "--regional-followup",
        action="store_true",
        help="Exact-distance disjoint subsampling and equal-search regional bias headroom",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Read-only, chunked per-layer regional attention measurement at sampled targets",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.regional_followup and (args.phase != "counterfactual" or args.attention):
        parser.error("--regional-followup requires counterfactual without --attention")
    if args.attention and args.phase != "counterfactual":
        parser.error("--attention is only supported for counterfactual measurements")
    if args.phase != "audit" and (not args.checkpoint or not args.sequence):
        parser.error("Model runs require --checkpoint and --sequence")
    if args.phase in {"diagnose", "counterfactual", "persistent", "baseline"}:
        if not args.validation_dir:
            parser.error("Diagnosis requires --validation-dir")
        validation = json.loads((args.validation_dir / "summary.json").read_text())
        if (
            validation.get("phase") != "validate"
            or not validation.get("completed")
            or not all(
                v["zero_controller"]["logits_bit_identical"]
                for v in validation["sequences"].values()
            )
        ):
            raise ValueError("Fresh MPS validation has not passed")
    if args.phase == "persistent":
        if not args.diagnosis_dir:
            parser.error("Persistent follow-up requires --diagnosis-dir")
        diagnosis = json.loads((args.diagnosis_dir / "summary.json").read_text())
        if not diagnosis.get("completed") or any(
            diagnosis["sequences"][video]["native"]["temporal"]["episode_count"] < 1
            for video in args.sequence
        ):
            raise ValueError("Completed B+ baseline with failure episodes is required")
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261018)
    setup = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "manifest_sha256": digest(args.manifest),
        "full_study_complete": False,
        "image_features": "Native per-state cache; independent persistent rollouts re-encode images. Counterfactuals restore target features before each replay.",
        "protocol_sha256": digest(ROOT / "protocols/region_tokens_v2_amendment.json"),
    }
    write_json(output / "setup.json", setup)
    for relative in [
        "scripts/run_region_v2_local.py",
        "modeling/region_tokens.py",
        "modeling/memory_attention.py",
        "modeling/memory_interventions.py",
        "protocols/region_tokens_v2_amendment.json",
    ]:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        f"Region v2 | local {args.phase}",
        "online",
        "https://forge.coreweave.com/api/wandb",
        setup,
        output,
        ["region-tokens-v2", "local", args.phase],
    )
    try:
        videos = (
            {
                **load_manifest(args.manifest, "dev"),
                **load_manifest(args.manifest, "test"),
            }
            if args.split == "all"
            else load_manifest(args.manifest, args.split)
        )
        if args.sequence:
            videos = {v: videos[v] for v in args.sequence}
        result = (
            audit(videos, output, run, args.side_matched)
            if args.phase == "audit"
            else counterfactual(videos, args, output, run)
            if args.phase == "counterfactual"
            else diagnose(videos, args, output, run)
        )
        write_json(output / "summary.json", result)
        run.summary["completed"] = result["completed"]
        log_results_artifact(
            run,
            output,
            ["summary.json", "setup.json"]
            + (
                ["feasibility.jsonl"]
                if args.phase == "audit"
                else ["frames.jsonl"]
                + (["attention.jsonl"] if args.attention else [])
                + (["headroom.jsonl"] if args.regional_followup else [])
                if args.phase == "counterfactual"
                else []
            ),
        )
    except Exception as error:
        write_json(output / "error.json", {"error": str(error), "completed": False})
        run.summary["completed"] = False
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
