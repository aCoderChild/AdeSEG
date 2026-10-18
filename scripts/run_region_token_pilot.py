#!/usr/bin/env python3
"""E0/E1 gate for region-token interventions, reusing the utility pipeline.

This entry point does not label an E1 pilot as a completed E2--E6 study.
Outputs live in the supplied remote/Drive directory; no default output path.
"""

# ruff: noqa: E402 -- allow direct CLI execution from any working directory

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "external/MedSAM2")]

from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from modeling.medsam2 import build_video_predictor
from modeling.region_tokens import (
    CATEGORIES,
    category_attention,
    indices_to_masks,
    indices_to_value_replacements,
    matched_experiments,
    region_memory_mask,
    token_categories,
)
from modeling.memory_attention import (
    UtilityController,
    attach_utility_controller,
)
from modeling.memory_interventions import replace_spatial_memory
from scripts.audit_region_initialization import prompt_index, target_indices
from training.tracking import start_wandb, log_results_artifact


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def finite(value):
    return float(value) if np.isfinite(value) else None


def capture_memory_read(module, inference):
    """Observe the native readout before SAM2 can gate final masks as absent."""
    reads = []
    handle = module.register_forward_hook(
        lambda _module, _inputs, output: reads.append(output.detach().clone())
    )
    try:
        result = inference()
    finally:
        handle.remove()
    if len(reads) != 1:
        raise RuntimeError(f"Expected one memory read, observed {len(reads)}")
    return result, reads[0]


def mutate_spatial_memory(bank):
    """Nonuniform fixed channel perturbation, on the independent test copy only."""
    for outputs in bank.values():
        for item in outputs.values():
            value = item.get("maskmem_features")
            if value is not None:
                if value.ndim != 4 or value.shape[1] < 2:
                    raise ValueError("Unexpected spatial-memory shape")
                value.add_(
                    torch.linspace(
                        -10, 10, value.shape[1], device=value.device, dtype=value.dtype
                    )[None, :, None, None]
                )


def memory_read_checks(reads):
    baseline = reads["baseline"]["read"]
    return {
        "noop_memory_reads_exact": all(
            torch.equal(baseline, reads[name]["read"])
            for name in ("bare", "empty", "zero", "value_noop", "memory_noop")
        ),
        "mutation_changed_memory_read": not torch.equal(
            baseline, reads["mutated"]["read"]
        ),
        "independent_memory_restore_exact": torch.equal(
            baseline, reads["restored"]["read"]
        ),
        "full_token_memory_equals_frame": torch.equal(
            reads["full_tokens"]["read"], reads["full_frame"]["read"]
        ),
    }


def selected_target_indices(frames, prompt, targets_per_sequence):
    """Select the frozen pilot grid or every eligible E2 target.

    ``None`` is deliberately the only all-targets value.  It avoids an
    ambiguous sentinel on the command line and keeps the existing E1 sampling
    untouched.
    """
    return target_indices(
        len(frames),
        prompt,
        len(frames) if targets_per_sequence is None else targets_per_sequence,
    )


def metrics(logits, target, void=None, decoder_logits=None):
    from monai.metrics import compute_hausdorff_distance

    prediction = (
        F.interpolate(
            logits.float(), size=target.shape, mode="bilinear", align_corners=False
        )[0, 0]
        .cpu()
        .numpy()
        > 0
    )
    valid = np.ones_like(target, dtype=bool) if void is None else ~void
    if not valid.any():
        return {
            "dice": None,
            "iou": None,
            "boundary_f": None,
            "hd95": None,
            "low_resolution_dice": None,
            "hd95_undefined": True,
        }
    values = calculate_scores(prediction[valid], target[valid])
    values = {key: values[key] for key in ("dice", "iou")}
    values["boundary_f"] = float(db_eval_boundary(target, prediction, void_pixels=void))
    prediction = prediction & valid
    target = target & valid
    if not target.any() and not prediction.any():
        values["hd95"] = 0.0
    elif not target.any() or not prediction.any():
        values["hd95"] = None
    else:
        values["hd95"] = finite(
            compute_hausdorff_distance(
                torch.from_numpy(prediction[None, None]),
                torch.from_numpy(target[None, None]),
                include_background=True,
                percentile=95,
            ).item()
        )
    values["hd95_undefined"] = values["hd95"] is None
    if void is not None and void.any():
        # MONAI has no void-aware surface-distance operator.
        values["hd95"] = None
        values["hd95_undefined"] = True
    if decoder_logits is False:
        values["low_resolution_dice"] = None
        return values
    if decoder_logits is not None:
        logits = decoder_logits
    low_target = (
        cv2.resize(
            target.astype(np.float32),
            logits.shape[-2:][::-1],
            interpolation=cv2.INTER_AREA,
        )
        >= 0.5
    )
    low_valid = (
        cv2.resize(
            valid.astype(np.float32),
            logits.shape[-2:][::-1],
            interpolation=cv2.INTER_AREA,
        )
        >= 1
    )
    values["low_resolution_dice"] = calculate_scores(
        (logits[0, 0].cpu().numpy() > 0)[low_valid], low_target[low_valid]
    )["dice"]
    return values


def logit_changes(baseline, altered, target, void=None):
    difference = (
        F.interpolate(
            (baseline - altered).abs().float(),
            target.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        .cpu()
        .numpy()
    )
    foreground = target.astype(np.uint8)
    outer = cv2.dilate(foreground, np.ones((11, 11), np.uint8)).astype(bool)
    inner = cv2.erode(foreground, np.ones((11, 11), np.uint8)).astype(bool)
    band = outer & ~inner
    if void is not None:
        band &= ~void
        inner &= ~void
    return {
        "boundary_band_logit_change": float(difference[band].mean())
        if band.any()
        else None,
        "interior_logit_change": float(difference[inner].mean())
        if inner.any()
        else None,
    }


def query_logit_changes(baseline, altered, queries):
    """Deployable query regions; save the 3 x memory-category effect matrix."""
    difference = (baseline - altered).abs().float()
    result = {}
    for name, mask in queries.items():
        side = int(np.sqrt(mask.shape[-1]))
        resized = F.interpolate(
            difference, (side, side), mode="bilinear", align_corners=False
        ).flatten(1)
        mask = mask.to(resized.device).bool()
        result[name] = float(resized[mask].mean()) if mask.any() else None
    return result


def query_dice_harm(baseline, altered, target, queries, void=None):
    """Signed overlap changes on the deployable 64x64 query regions."""
    side = int(np.sqrt(next(iter(queries.values())).shape[-1]))
    truth = (
        cv2.resize(
            target.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA
        ).ravel()
        >= 0.5
    )
    predictions = [
        (
            F.interpolate(
                value.float(), (side, side), mode="bilinear", align_corners=False
            )[0, 0]
            .cpu()
            .numpy()
            .ravel()
            > 0
        )
        for value in (baseline, altered)
    ]
    result = {}
    valid = (
        np.ones(side * side, dtype=bool)
        if void is None
        else cv2.resize(
            (~void).astype(np.float32), (side, side), interpolation=cv2.INTER_AREA
        ).ravel()
        >= 1
    )
    for name, query in queries.items():
        selected = query[0].cpu().numpy().astype(bool) & valid
        result[name] = (
            (
                calculate_scores(predictions[0][selected], truth[selected])["dice"]
                - calculate_scores(predictions[1][selected], truth[selected])["dice"]
            )
            if selected.any()
            else None
        )
    return result


def swap_replacements(chosen, selected, labels, donors, video, relative_time):
    """Same-category donor tokens at the exact same 64x64 spatial position.

    Try other sequences in temporal-distance order. If any selected token has
    no eligible donor, skip the entire comparison without reducing its budget.
    """
    local = np.asarray(selected) % 4096
    replacements = np.empty((len(selected), 64), np.float32)
    pending = np.ones(len(selected), bool)
    used = set()
    candidates = sorted(
        (d for d in donors if d["video"] != video),
        key=lambda d: (abs(d["relative_time"] - relative_time), d["path"]),
    )
    for donor in candidates:
        with np.load(donor["path"], allow_pickle=False) as data:
            available = pending & (
                data["categories"].ravel()[local] == labels[selected]
            )
            if available.any():
                replacements[available] = data["values"][local[available]]
                pending[available] = False
                used.add(donor["video"])
        if not pending.any():
            return indices_to_value_replacements(
                chosen, selected, replacements
            ), sorted(used)
    return None, sorted(used)


@torch.inference_mode()
def run_sequence(
    predictor,
    controller,
    video,
    frames,
    protocol,
    output,
    run,
    e0_only,
    value_calibration=None,
    e2_mean_smoke=False,
    mean_values=None,
    targets_per_sequence=None,
    experiment=None,
    e0_target_frame=None,
):
    """Natural pointers, c0 spatial memory, independent copies per counterfactual."""
    checks, rows = [], 0
    experiment = experiment or {}
    condition = experiment.get("memory_condition", "c0")
    label_id = experiment.get("label_id", 1)
    label_ids = experiment.get("label_ids", [1])

    def gt(index):
        return load_label_mask(frames[index].mask_path, label_ids) == label_id

    prompt = prompt_index(frames, protocol, label_id, label_ids)
    first_mask = gt(prompt)
    donors = experiment.get("donors", [])
    targets = selected_target_indices(frames, prompt, targets_per_sequence)
    if e0_target_frame is not None:
        if not e0_only or e0_target_frame not in selected_target_indices(
            frames, prompt, None
        ):
            raise ValueError("An explicit E0 target must be eligible and sanity-only")
        targets = [e0_target_frame]
    if e2_mean_smoke:
        targets = targets[:1]
    targets = set(targets)
    write_json(
        output / f"initialization_{video}.json",
        {"video": video, "prompt_frame": prompt, "target_frames": sorted(targets)},
    )
    with tempfile.TemporaryDirectory(prefix="region_frames_") as directory:
        stage_frames(frames, Path(directory))
        state = predictor.init_state(
            video_path=directory, offload_video_to_cpu=True, offload_state_to_cpu=True
        )
        predictor.add_new_mask(
            state, frame_idx=prompt, obj_id=1, mask=torch.from_numpy(first_mask)
        )
        taxonomy = {}
        try:
            for t, _, _ in predictor.propagate_in_video(state, start_frame_idx=prompt):
                if t in targets:
                    # Capture the bank strictly before t. The yielded current prediction
                    # was not generated by an intervention and is not in this snapshot.
                    bank = copy.deepcopy(state["output_dict"])
                    bank["non_cond_frame_outputs"].pop(t, None)
                    cached = copy.deepcopy(state["cached_features"])

                    def replay(
                        token_masks=None,
                        excluded_frames=(),
                        token_biases=None,
                        token_value_replacements=None,
                        token_memory_replacements=None,
                        pointers=False,
                        read_trace=None,
                    ):
                        state["cached_features"] = copy.deepcopy(cached)
                        controller.excluded_token_masks = token_masks or {}
                        controller.excluded_frame_indices = set(excluded_frames)
                        controller.token_biases = token_biases or {}
                        controller.token_value_replacements = (
                            token_value_replacements or {}
                        )
                        controller.token_memory_replacements = (
                            token_memory_replacements or {}
                        )
                        controller.exclude_pointers = pointers

                        def inference():
                            return predictor.run_single_frame_inference(
                                inference_state=state,
                                output_dict=copy.deepcopy(bank),
                                frame_idx=t,
                                batch_size=1,
                                is_init_cond_frame=False,
                                point_inputs=None,
                                mask_inputs=None,
                                reverse=False,
                                run_mem_encoder=False,
                            )

                        if read_trace is not None:
                            result, read = capture_memory_read(
                                predictor.memory_attention, inference
                            )
                            read_trace.update(
                                read=read,
                                object_score_logits=result[0]["object_score_logits"]
                                .detach()
                                .clone(),
                            )
                        else:
                            result = inference()
                        return result[1].detach().clone()

                    reads = {}

                    def checked_replay(name, **kwargs):
                        reads[name] = {}
                        return replay(read_trace=reads[name], **kwargs)

                    predictor.memory_attention.capture_attention = True
                    predictor.memory_attention.capture_memory_values = True
                    baseline = checked_replay("baseline")
                    trace = predictor.memory_attention.last_trace
                    units = trace["units"]
                    if len(units) != 7 or any(
                        u["end"] - u["start"] != 4096 for u in units
                    ):
                        raise RuntimeError(
                            f"Unexpected default memory layout at {video}:{t}: {units}"
                        )
                    reconstruction = max(trace["reconstruction_max_abs"])
                    captured_memory_values = trace["memory_values"].detach().cpu()
                    value_identity = {
                        unit["frame_idx"]: (
                            np.ones(unit["end"] - unit["start"], dtype=bool),
                            trace["memory_values"][0, unit["start"] : unit["end"]],
                        )
                        for unit in units
                    }
                    pointer_count = trace["attention"][0].shape[-1] - max(
                        u["end"] for u in units
                    )
                    attention = {
                        name: category_attention(
                            trace["attention"],
                            units,
                            taxonomy,
                            trace["query_regions"],
                            conditioning=condition,
                        )
                        for name, condition in (
                            ("conditioning", True),
                            ("recent", False),
                        )
                    }
                    queries = trace["query_regions"]
                    # Anchor similarity uses the official Lean-SAM2 max-cosine
                    # implementation, imported only when E5 is requested.
                    anchor_scores = None
                    if experiment.get("anchor_similarity"):
                        from scripts.run_region_experiments import (
                            lean_anchor_similarity,
                        )

                        anchor_scores = lean_anchor_similarity(
                            captured_memory_values, units, taxonomy
                        )
                    predictor.memory_attention.capture_attention = False
                    predictor.memory_attention.capture_memory_values = False
                    # Drop the multi-GB raw attention tensors before counterfactuals.
                    trace["attention"] = []
                    del trace
                    for layer in predictor.memory_attention.layers:
                        layer.cross_attn_image.last_attention = None
                    predictor.memory_attention.utility_controller = None
                    bare = checked_replay("bare")
                    predictor.memory_attention.utility_controller = controller
                    empty = checked_replay(
                        "empty",
                        token_masks={
                            u["frame_idx"]: np.zeros(4096, bool) for u in units
                        },
                    )
                    zero = checked_replay(
                        "zero",
                        token_biases={u["frame_idx"]: np.zeros(4096) for u in units},
                    )
                    value_noop = checked_replay(
                        "value_noop", token_value_replacements=value_identity
                    )
                    memory_noop = checked_replay(
                        "memory_noop", token_memory_replacements=value_identity
                    )
                    # Independent restore: mutate the bank, run from the mutation, then
                    # restore a separate saved copy. This is not a duplicated no-op.
                    saved = copy.deepcopy(bank)
                    legacy_diagnostic = {}
                    if e0_target_frame is not None:
                        for outputs in bank.values():
                            for item in outputs.values():
                                if item.get("maskmem_features") is not None:
                                    item["maskmem_features"].add_(10)
                        legacy_prediction = checked_replay("legacy_uniform_mutated")
                        legacy_diagnostic = {
                            "legacy_uniform_mutation_changed_prediction": not torch.equal(
                                baseline, legacy_prediction
                            ),
                            "legacy_uniform_mutation_changed_memory_read": not torch.equal(
                                reads["baseline"]["read"],
                                reads["legacy_uniform_mutated"]["read"],
                            ),
                            "legacy_uniform_object_score_logits": reads[
                                "legacy_uniform_mutated"
                            ]["object_score_logits"]
                            .float()
                            .cpu()
                            .flatten()
                            .tolist(),
                        }
                        bank = copy.deepcopy(saved)
                    mutate_spatial_memory(bank)
                    mutated = checked_replay("mutated")
                    bank = copy.deepcopy(saved)
                    restored = checked_replay("restored")
                    unit = units[0]
                    full_tokens = checked_replay(
                        "full_tokens",
                        token_masks={unit["frame_idx"]: np.ones(4096, bool)},
                    )
                    full_frame = checked_replay(
                        "full_frame", excluded_frames=[unit["frame_idx"]]
                    )
                    check = {
                        "restore_control_version": 2,
                        "video": video,
                        "frame": int(t),
                        "detached_noop_exact": torch.equal(bare, baseline),
                        "empty_token_mask_exact": torch.equal(bare, empty),
                        "zero_bias_exact": torch.equal(bare, zero),
                        "value_identity_exact": torch.equal(bare, value_noop),
                        "memory_identity_exact": torch.equal(bare, memory_noop),
                        "independent_restore_exact": torch.equal(baseline, restored),
                        "mutation_changed_prediction": not torch.equal(
                            baseline, mutated
                        ),
                        "full_token_equals_frame": torch.equal(full_tokens, full_frame),
                        "attention_reconstruction_max_abs": float(reconstruction),
                        **memory_read_checks(reads),
                        "baseline_object_score_logits": reads["baseline"][
                            "object_score_logits"
                        ]
                        .float()
                        .cpu()
                        .flatten()
                        .tolist(),
                        "mutated_object_score_logits": reads["mutated"][
                            "object_score_logits"
                        ]
                        .float()
                        .cpu()
                        .flatten()
                        .tolist(),
                        "baseline_logits_constant": bool(
                            torch.all(baseline == baseline.flatten()[0])
                        ),
                        "baseline_constant_value": float(baseline.flatten()[0])
                        if bool(torch.all(baseline == baseline.flatten()[0]))
                        else None,
                        **legacy_diagnostic,
                        "mutation_prediction_check_role": "diagnostic only; native absence gating can hide a changed memory read",
                    }
                    check["passed"] = (
                        all(
                            v
                            for k, v in check.items()
                            if k.endswith("exact")
                            or k
                            in {
                                "full_token_equals_frame",
                                "mutation_changed_memory_read",
                                "full_token_memory_equals_frame",
                            }
                        )
                        and reconstruction < 1e-5
                    )
                    checks.append(check)
                    del reads
                    with (output / "sanity.jsonl").open("a") as stream:
                        stream.write(json.dumps(check) + "\n")
                    run.log(
                        {
                            "E0/passed": int(check["passed"]),
                            "E0/reconstruction_error": reconstruction,
                        }
                    )
                    if not check["passed"]:
                        raise RuntimeError(f"E0 failed: {check}")
                    if e0_only:
                        return checks, rows
                    if value_calibration is not None:
                        values = captured_memory_values[0].float()
                        for unit in units:
                            labels = taxonomy[unit["frame_idx"]].ravel()
                            spatial = values[unit["start"] : unit["end"]]
                            for category, name in enumerate(CATEGORIES):
                                selected = labels == category
                                if selected.any():
                                    value_calibration[name]["sum"] += (
                                        spatial[selected].sum(0).numpy()
                                    )
                                    value_calibration[name]["count"] += int(
                                        selected.sum()
                                    )
                            if experiment.get("save_donors"):
                                path = (
                                    output
                                    / f"donor_{video}_object{label_id}_{unit['frame_idx']}.npz"
                                )
                                if not path.exists():
                                    np.savez_compressed(
                                        path,
                                        values=spatial.numpy(),
                                        categories=taxonomy[unit["frame_idx"]],
                                        video=video,
                                        label_id=label_id,
                                        relative_time=unit["frame_idx"]
                                        / max(1, len(frames) - 1),
                                    )
                        controller.excluded_token_masks = {}
                        controller.excluded_frame_indices = set()
                        controller.token_biases = {}
                        controller.token_value_replacements = {}
                        controller.exclude_pointers = False
                        replace_spatial_memory(
                            predictor,
                            state,
                            state["output_dict"],
                            frames,
                            t,
                            label_id,
                            label_ids,
                            "ground_truth",
                            0,
                            "mixed",
                        )
                        taxonomy[t] = token_categories(
                            gt(t),
                            band=experiment.get("band", 1),
                            thresholds=experiment.get("thresholds", (0.1, 0.9)),
                            boundary_definition=experiment.get("boundary_definition", "mixed_coverage"),
                        )["categories"]
                        write_json(
                            output / "progress.json",
                            {
                                "state": "running",
                                "video": video,
                                "frame": int(t),
                                "rows": rows,
                            },
                        )
                        continue
                    target = gt(t)
                    void = np.isin(
                        load_label_mask(frames[t].mask_path, label_ids),
                        experiment.get("ignore_labels", []),
                    )
                    base_metrics = metrics(baseline, target, void)
                    if e2_mean_smoke:
                        for bank_name, conditioning in (
                            ("conditioning", True),
                            ("recent", False),
                        ):
                            chosen = [
                                u
                                for u in units
                                if bool(u["conditioning"]) == conditioning
                            ]
                            labels = np.concatenate(
                                [taxonomy[u["frame_idx"]].ravel() for u in chosen]
                            )
                            selected = np.flatnonzero(
                                labels == CATEGORIES.index("boundary")
                            )
                            k = len(selected)
                            row = {
                                "video": video,
                                "frame": int(t),
                                "bank": bank_name,
                                "category": "boundary",
                                "k": k,
                                "memory_source": "c0",
                                "mode": "single_step",
                                "baseline": base_metrics,
                            }
                            if not k:
                                row.update(
                                    state="skipped", reason="empty_boundary_tokens"
                                )
                            else:
                                replacement = np.repeat(
                                    np.asarray(mean_values["boundary"])[None], k, axis=0
                                )
                                results = {
                                    "attention_mask": replay(
                                        indices_to_masks(chosen, selected)
                                    ),
                                    "mean_substitution": replay(
                                        token_value_replacements=indices_to_value_replacements(
                                            chosen, selected, replacement
                                        )
                                    ),
                                }
                                row["state"] = "complete"
                                row["interventions"] = {
                                    name: metrics(logits, target)
                                    for name, logits in results.items()
                                }
                            with (output / "mean_smoke.jsonl").open("a") as stream:
                                stream.write(json.dumps(row, allow_nan=False) + "\n")
                            rows += 1
                        return checks, rows
                    for name, altered, removed in (
                        (
                            "all_spatial",
                            replay(excluded_frames=[u["frame_idx"] for u in units]),
                            4096 * len(units),
                        ),
                        ("pointers_only", replay(pointers=True), pointer_count),
                    ):
                        with (output / "bounds.jsonl").open("a") as stream:
                            stream.write(
                                json.dumps(
                                    {
                                        "video": video,
                                        "frame": int(t),
                                        "control": name,
                                        "removed_tokens": removed,
                                        "baseline": base_metrics,
                                        "intervention": metrics(altered, target, void),
                                    },
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                    for bank_name, conditioning in (
                        ("conditioning", True),
                        ("recent", False),
                    ):
                        chosen = [
                            u for u in units if bool(u["conditioning"]) == conditioning
                        ]
                        labels = np.concatenate(
                            [taxonomy[u["frame_idx"]].ravel() for u in chosen]
                        )
                        k, reference, experiments = matched_experiments(
                            labels.reshape(-1, 64, 64), protocol["budget"]["seeds"],
                            spatial_match=experiment.get("spatial_match", "pooled_distance"),
                        )
                        intervention_kind = experiment.get(
                            "intervention", "attention_mask"
                        )
                        memory_dice = []
                        for u in chosen:
                            truth = gt(u["frame_idx"])
                            if condition == "natural":
                                stored = bank[
                                    "cond_frame_outputs"
                                    if u["conditioning"]
                                    else "non_cond_frame_outputs"
                                ][u["frame_idx"]]["pred_masks"]
                                encoded_mask = (
                                    F.interpolate(
                                        stored.float(),
                                        truth.shape,
                                        mode="bilinear",
                                        align_corners=False,
                                    )[0, 0]
                                    .cpu()
                                    .numpy()
                                    > 0
                                )
                            else:
                                encoded_mask = region_memory_mask(
                                    truth, condition, u["frame_idx"], u["conditioning"]
                                )
                                encoded_mask = cv2.resize(
                                    encoded_mask.astype(np.uint8),
                                    truth.shape[::-1],
                                    interpolation=cv2.INTER_NEAREST,
                                ).astype(bool)
                            memory_dice.append(
                                calculate_scores(encoded_mask, truth)["dice"]
                            )
                        for category, seed, selected in experiments:
                            row = {
                                "video": video,
                                "frame": int(t),
                                "bank": bank_name,
                                "category": category,
                                "seed": seed,
                                "k": k,
                                "memory_source": condition,
                                "mean_memory_mask_dice": float(np.mean(memory_dice)),
                                "intervention_kind": intervention_kind,
                                "dataset": experiment.get("dataset", "polypgen"),
                                "category_source": experiment.get(
                                    "category_source", "oracle"
                                ),
                                "source_video": experiment.get("source_video", video),
                                "split": experiment.get("split", "test"),
                                "label_id": label_id,
                                "band": experiment.get("band", 1),
                                "thresholds": experiment.get("thresholds", [0.1, 0.9]),
                                "mode": "single_step",
                                "boundary_definition": experiment.get("boundary_definition", "mixed_coverage"),
                                "spatial_match": experiment.get("spatial_match", "pooled_distance"),
                                "baseline": base_metrics,
                                "attention": attention[bank_name],
                                "anchor_similarity": anchor_scores.get(bank_name)
                                if anchor_scores
                                else None,
                                "category_counts": dict(
                                    zip(
                                        CATEGORIES,
                                        np.bincount(labels, minlength=4).tolist(),
                                    )
                                ),
                            }
                            if void.all():
                                row.update(
                                    state="skipped", reason="fully_unannotated_target"
                                )
                            elif selected is None:
                                row.update(
                                    state="skipped",
                                    reason="empty_boundary_or_insufficient_category_tokens",
                                )
                            else:
                                if intervention_kind == "attention_mask":
                                    altered = replay(indices_to_masks(chosen, selected))
                                elif intervention_kind == "mean_substitution":
                                    means = np.asarray(
                                        [
                                            mean_values[CATEGORIES[c]]
                                            for c in labels[selected]
                                        ]
                                    )
                                    altered = replay(
                                        token_value_replacements=indices_to_value_replacements(
                                            chosen, selected, means
                                        )
                                    )
                                elif intervention_kind == "swap":
                                    replacement, donor_videos = swap_replacements(
                                        chosen,
                                        selected,
                                        labels,
                                        donors,
                                        video,
                                        t / max(1, len(frames) - 1),
                                    )
                                    if replacement is None:
                                        row.update(
                                            state="skipped",
                                            reason="no_cross_sequence_category_donor",
                                        )
                                        with (output / "frames.jsonl").open(
                                            "a"
                                        ) as stream:
                                            stream.write(
                                                json.dumps(row, allow_nan=False) + "\n"
                                            )
                                        rows += 1
                                        continue
                                    row.update(
                                        donor_videos=donor_videos,
                                        donor_position_distance=0.0,
                                    )
                                    altered = replay(
                                        token_memory_replacements=replacement
                                    )
                                else:
                                    raise ValueError(intervention_kind)
                                scores = metrics(altered, target, void)
                                effects = {
                                    name: (base_metrics[name] - scores[name])
                                    if base_metrics[name] is not None
                                    and scores[name] is not None
                                    else None
                                    for name in (
                                        "dice",
                                        "iou",
                                        "boundary_f",
                                        "hd95",
                                        "low_resolution_dice",
                                    )
                                }
                                # HD95 harm has the opposite sign to overlap-score harm.
                                effects["hd95"] = (
                                    -effects["hd95"]
                                    if effects["hd95"] is not None
                                    else None
                                )
                                row.update(
                                    state="complete",
                                    intervention=scores,
                                    harm=effects,
                                    harm_per_token={
                                        name: value / k if value is not None else None
                                        for name, value in effects.items()
                                    },
                                    reference_overlap=float(
                                        np.isin(selected, reference).mean()
                                    ),
                                    **logit_changes(baseline, altered, target, void),
                                    query_logit_changes=query_logit_changes(
                                        baseline, altered, queries
                                    ),
                                    query_dice_harm=query_dice_harm(
                                        baseline, altered, target, queries, void
                                    ),
                                )
                                run.log(
                                    {
                                        f"{experiment.get('stage', 'E1')}/{bank_name}/{category}/{name}": value
                                        for name, value in effects.items()
                                        if value is not None
                                    }
                                )
                            with (output / "frames.jsonl").open("a") as stream:
                                stream.write(json.dumps(row, allow_nan=False) + "\n")
                                stream.flush()
                            rows += 1
                    print(
                        f"[region] {video} target={t} completed_rows={rows}", flush=True
                    )
                    write_json(
                        output / "progress.json",
                        {
                            "state": "running",
                            "video": video,
                            "frame": int(t),
                            "rows": rows,
                        },
                    )
                controller.excluded_token_masks = {}
                controller.excluded_frame_indices = set()
                controller.token_biases = {}
                controller.token_value_replacements = {}
                controller.token_memory_replacements = {}
                controller.exclude_pointers = False
                # Preserve the natural pointer, replace only spatial memory.
                replace_spatial_memory(
                    predictor,
                    state,
                    state["output_dict"],
                    frames,
                    t,
                    label_id,
                    label_ids,
                    "predicted" if condition == "natural" else "ground_truth",
                    0,
                    "mixed",
                    mask_override=None
                    if condition == "natural"
                    else region_memory_mask(gt(t), condition, t, t == prompt),
                )
                category_mask = gt(t)
                if experiment.get("category_source") == "deployable":
                    category_mask = (
                        state["output_dict"][
                            "cond_frame_outputs"
                            if t == prompt
                            else "non_cond_frame_outputs"
                        ][t]["pred_masks"][0, 0]
                        .float()
                        .cpu()
                        .numpy()
                        > 0
                    )
                taxonomy[t] = token_categories(
                    category_mask,
                    band=experiment.get("band", 1),
                    thresholds=experiment.get("thresholds", (0.1, 0.9)),
                    boundary_definition=experiment.get("boundary_definition", "mixed_coverage"),
                )["categories"]
        finally:
            controller.excluded_token_masks = {}
            controller.excluded_frame_indices = set()
            controller.token_biases = {}
            controller.token_value_replacements = {}
            controller.token_memory_replacements = {}
            controller.exclude_pointers = False
            predictor.reset_state(state)
    return checks, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "protocols/region_tokens_v1_1.json"
    )
    parser.add_argument(
        "--e2-mean-smoke",
        action="store_true",
        help="Run one preregistered target per dev sequence to validate non-identity mean substitution.",
    )
    parser.add_argument(
        "--value-calibration",
        type=Path,
        help="Completed E2_calibration/value_calibration.json used only with --e2-mean-smoke.",
    )
    parser.add_argument("--e0-only", action="store_true")
    parser.add_argument(
        "--sequence",
        action="append",
        help="Repeat to select an explicit, disjoint resumable sequence shard.",
    )
    parser.add_argument(
        "--all-eligible-targets",
        action="store_true",
        help="Use every target from prompt + 7 onward; only valid for the E2 attention component.",
    )
    parser.add_argument(
        "--stage",
        choices=("E1", "E2_attention_primary"),
        default="E1",
        help="Result label. E2_attention_primary is only the masking component of E2.",
    )
    parser.add_argument(
        "--calibrate-values",
        action="store_true",
        help="Record dev-only c0 category value means after E0; do not run E1 effects.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--e0-target-frame",
        type=int,
        help="Diagnostic E0 target only; never changes experimental sampling.",
    )
    args = parser.parse_args()
    if args.e0_target_frame is not None and not args.e0_only:
        parser.error("--e0-target-frame requires --e0-only")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    videos = load_manifest(args.manifest, "test")
    selected_modes = (
        int(args.e0_only) + int(args.calibrate_values) + int(args.e2_mean_smoke)
    )
    if selected_modes > 1:
        raise ValueError(
            "--e0-only, --calibrate-values and --e2-mean-smoke are mutually exclusive"
        )
    if args.e2_mean_smoke != bool(args.value_calibration):
        raise ValueError("--e2-mean-smoke requires --value-calibration and vice versa")
    mean_values = None
    if args.value_calibration:
        calibration = json.loads(args.value_calibration.read_text())
        mean_values = {
            name: values["mean"] for name, values in calibration["categories"].items()
        }
        if set(mean_values) != set(CATEGORIES) or any(
            len(values) != 64 or not np.isfinite(values).all()
            for values in mean_values.values()
        ):
            raise ValueError(
                "Value calibration must contain finite 64-D means for every category"
            )
    requested = args.sequence or protocol["pilot"]["sequences"]
    if args.e0_only:
        requested = requested[:1]
    if args.all_eligible_targets and args.stage != "E2_attention_primary":
        raise ValueError("--all-eligible-targets requires --stage E2_attention_primary")
    if args.stage == "E2_attention_primary" and not args.all_eligible_targets:
        raise ValueError("E2 attention component requires --all-eligible-targets")
    if args.stage != "E1" and (
        args.e0_only or args.calibrate_values or args.e2_mean_smoke
    ):
        raise ValueError(
            "E2 attention component cannot be combined with a calibration or smoke mode"
        )
    if set(requested) - videos.keys():
        raise ValueError("Missing pilot sequences")
    torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    setup = {
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sequences": requested,
        "e0_target_frame": args.e0_target_frame,
        "torch": torch.__version__,
        "stage": (
            "E0"
            if args.e0_only
            else "E2_calibration"
            if args.calibrate_values
            else "E2_mean_smoke"
            if args.e2_mean_smoke
            else args.stage
        ),
        "scope": (
            "dev-only c0 value calibration; E2 effects not executed"
            if args.calibrate_values
            else "dev-only E2 mean-value substitution operator smoke; no E2 effect claim"
            if args.e2_mean_smoke
            else "E2 c0 attention-masking component; mean substitution, swap, persistent, and cross-domain components are not executed"
            if args.stage == "E2_attention_primary"
            else "gated region pilot; E2-E6 not executed"
        ),
        "polypgen_mirror_equivalence": "not checksum-verified against Synapse",
    }
    write_json(args.output_dir / "setup.json", setup)
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        f"region-tokens-{setup['stage']}",
        "online",
        "https://forge.coreweave.com/api/wandb",
        setup,
        args.output_dir,
        ["region-tokens-v1", setup["stage"]],
    )
    started = time.time()
    try:
        predictor = build_video_predictor(
            protocol["model"]["config"], args.checkpoint, args.device
        ).eval()
        controller = UtilityController()
        controller.enabled = False
        attach_utility_controller(predictor, controller)
        checks, total = [], 0
        calibration = (
            {
                name: {
                    "sum": np.zeros(predictor.mem_dim, dtype=np.float64),
                    "count": 0,
                }
                for name in CATEGORIES
            }
            if args.calibrate_values
            else None
        )
        for video in requested:
            print(f"[region] sequence={video} starting", flush=True)
            sanity, rows = run_sequence(
                predictor,
                controller,
                video,
                videos[video],
                protocol,
                args.output_dir,
                run,
                args.e0_only,
                calibration,
                args.e2_mean_smoke,
                mean_values,
                None
                if args.all_eligible_targets
                else protocol["pilot"]["targets_per_sequence"],
                e0_target_frame=args.e0_target_frame,
            )
            checks.extend(sanity)
            total += rows
            run.log(
                {
                    "progress/sequences": len(set(c["video"] for c in checks)),
                    "progress/rows": total,
                    "cost/gpu_hours": (time.time() - started) / 3600,
                }
            )
            log_results_artifact(
                run, args.output_dir, [p.name for p in args.output_dir.glob("*.json*")]
            )
        if calibration is not None:
            means = {
                name: {
                    "count": item["count"],
                    "mean": (item["sum"] / item["count"]).tolist()
                    if item["count"]
                    else None,
                }
                for name, item in calibration.items()
            }
            if any(value["mean"] is None for value in means.values()):
                raise RuntimeError("A preregistered category has no calibration values")
            write_json(
                args.output_dir / "value_calibration.json",
                {
                    "source": "PolypGen dev pilot sequences; c0 spatial memory; E0 passed each target",
                    "categories": means,
                },
            )
        write_json(
            args.output_dir / "status.json",
            {
                "state": "complete",
                "stage": setup["stage"],
                "e0_pass": bool(checks) and all(c["passed"] for c in checks),
                "sequences": requested,
                "rows": total,
                "elapsed_seconds": time.time() - started,
            },
        )
        log_results_artifact(
            run, args.output_dir, [p.name for p in args.output_dir.glob("*.json*")]
        )
    except BaseException as exc:
        write_json(
            args.output_dir / "status.json",
            {
                "state": "failed",
                "error": str(exc),
                "elapsed_seconds": time.time() - started,
            },
        )
        log_results_artifact(
            run, args.output_dir, [p.name for p in args.output_dir.glob("*.json*")]
        )
        run.finish(exit_code=1)
        raise
    run.finish()


if __name__ == "__main__":
    main()
