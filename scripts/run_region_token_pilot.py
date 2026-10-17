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

from adenoid.io import load_label_mask
from datasets.video import load_manifest
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from modeling.medsam2 import build_video_predictor
from modeling.region_tokens import (
    CATEGORIES,
    category_attention,
    indices_to_masks,
    matched_indices,
    spatially_matched_indices,
    token_categories,
)
from modeling.utility_memory import (
    UtilityController,
    UtilityHead,
    attach_utility_controller,
)
from scripts.generate_utility_labels import replace_spatial_memory
from scripts.infer import stage_frames
from scripts.audit_region_initialization import prompt_index, target_indices
from training.tracking import start_wandb, log_results_artifact


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def finite(value):
    return float(value) if np.isfinite(value) else None


def metrics(logits, target):
    from monai.metrics import compute_hausdorff_distance

    prediction = (
        F.interpolate(
            logits.float(), size=target.shape, mode="bilinear", align_corners=False
        )[0, 0]
        .cpu()
        .numpy()
        > 0
    )
    values = calculate_scores(prediction, target)
    values = {key: values[key] for key in ("dice", "iou")}
    values["boundary_f"] = float(db_eval_boundary(target, prediction))
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
    low_target = (
        cv2.resize(
            target.astype(np.float32),
            logits.shape[-2:][::-1],
            interpolation=cv2.INTER_AREA,
        )
        >= 0.5
    )
    values["low_resolution_dice"] = calculate_scores(
        logits[0, 0].cpu().numpy() > 0, low_target
    )["dice"]
    return values


def logit_changes(baseline, altered, target):
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
    return {
        "boundary_band_logit_change": float(difference[band].mean())
        if band.any()
        else None,
        "interior_logit_change": float(difference[inner].mean())
        if inner.any()
        else None,
    }


@torch.inference_mode()
def run_sequence(predictor, controller, video, frames, protocol, output, run, e0_only):
    """Natural pointers, c0 spatial memory, independent copies per counterfactual."""
    checks, rows = [], 0
    prompt = prompt_index(frames, protocol)
    first_mask = load_label_mask(frames[prompt].mask_path, [1]) == 1
    targets = set(
        target_indices(len(frames), prompt, protocol["pilot"]["targets_per_sequence"])
    )
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
                        pointers=False,
                    ):
                        state["cached_features"] = copy.deepcopy(cached)
                        controller.excluded_token_masks = token_masks or {}
                        controller.excluded_frame_indices = set(excluded_frames)
                        controller.token_biases = token_biases or {}
                        controller.exclude_pointers = pointers
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

                    predictor.memory_attention.capture_attention = True
                    baseline = replay()
                    trace = predictor.memory_attention.last_trace
                    units = trace["units"]
                    if len(units) != 7 or any(
                        u["end"] - u["start"] != 4096 for u in units
                    ):
                        raise RuntimeError(
                            f"Unexpected default memory layout at {video}:{t}: {units}"
                        )
                    reconstruction = max(trace["reconstruction_max_abs"])
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
                    predictor.memory_attention.capture_attention = False
                    # Drop the multi-GB raw attention tensors before counterfactuals.
                    trace["attention"] = []
                    del trace
                    for layer in predictor.memory_attention.layers:
                        layer.cross_attn_image.last_attention = None
                    predictor.memory_attention.utility_controller = None
                    bare = replay()
                    predictor.memory_attention.utility_controller = controller
                    empty = replay(
                        {u["frame_idx"]: np.zeros(4096, bool) for u in units}
                    )
                    zero = replay(
                        token_biases={u["frame_idx"]: np.zeros(4096) for u in units}
                    )
                    # Independent restore: mutate the bank, run from the mutation, then
                    # restore a separate saved copy. This is not a duplicated no-op.
                    saved = copy.deepcopy(bank)
                    for outputs in bank.values():
                        for item in outputs.values():
                            if item.get("maskmem_features") is not None:
                                item["maskmem_features"].add_(10)
                    mutated = replay()
                    bank = copy.deepcopy(saved)
                    restored = replay()
                    unit = units[0]
                    full_tokens = replay({unit["frame_idx"]: np.ones(4096, bool)})
                    full_frame = replay(excluded_frames=[unit["frame_idx"]])
                    check = {
                        "video": video,
                        "frame": int(t),
                        "detached_noop_exact": torch.equal(bare, baseline),
                        "empty_token_mask_exact": torch.equal(bare, empty),
                        "zero_bias_exact": torch.equal(bare, zero),
                        "independent_restore_exact": torch.equal(baseline, restored),
                        "mutation_changed_prediction": not torch.equal(
                            baseline, mutated
                        ),
                        "full_token_equals_frame": torch.equal(full_tokens, full_frame),
                        "attention_reconstruction_max_abs": float(reconstruction),
                    }
                    check["passed"] = (
                        all(
                            v
                            for k, v in check.items()
                            if k.endswith("exact")
                            or k
                            in {
                                "full_token_equals_frame",
                                "mutation_changed_prediction",
                            }
                        )
                        and reconstruction < 1e-5
                    )
                    checks.append(check)
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
                    target = load_label_mask(frames[t].mask_path, [1]) == 1
                    base_metrics = metrics(baseline, target)
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
                                        "intervention": metrics(altered, target),
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
                        k = int((labels == 0).sum())
                        reference = np.flatnonzero(labels == 0)
                        experiments = [
                            (name, 0, matched_indices(labels, i, k, 0))
                            for i, name in enumerate(CATEGORIES)
                        ]
                        for seed in protocol["budget"]["seeds"]:
                            uniform = (
                                np.random.default_rng(seed).choice(
                                    len(labels), k, replace=False
                                )
                                if k
                                else None
                            )
                            spatial = (
                                spatially_matched_indices(
                                    labels.reshape(-1, 64, 64), reference, seed
                                )
                                if k
                                else None
                            )
                            experiments.extend(
                                [
                                    ("random", seed, uniform),
                                    ("spatial_random", seed, spatial),
                                ]
                            )
                        for category, seed, selected in experiments:
                            row = {
                                "video": video,
                                "frame": int(t),
                                "bank": bank_name,
                                "category": category,
                                "seed": seed,
                                "k": k,
                                "memory_source": "c0",
                                "mode": "single_step",
                                "baseline": base_metrics,
                                "attention": attention[bank_name],
                                "category_counts": dict(
                                    zip(
                                        CATEGORIES,
                                        np.bincount(labels, minlength=4).tolist(),
                                    )
                                ),
                            }
                            if selected is None:
                                row.update(
                                    state="skipped",
                                    reason="empty_boundary_or_insufficient_category_tokens",
                                )
                            else:
                                altered = replay(indices_to_masks(chosen, selected))
                                scores = metrics(altered, target)
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
                                    **logit_changes(baseline, altered, target),
                                )
                                run.log(
                                    {
                                        f"E1/{bank_name}/{category}/{name}": value
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
                controller.exclude_pointers = False
                # Preserve the natural pointer, replace only spatial memory.
                replace_spatial_memory(
                    predictor,
                    state,
                    state["output_dict"],
                    frames,
                    t,
                    1,
                    [1],
                    "ground_truth",
                    0,
                    "mixed",
                )
                taxonomy[t] = token_categories(
                    load_label_mask(frames[t].mask_path, [1]) == 1
                )["categories"]
        finally:
            controller.excluded_token_masks = {}
            controller.excluded_frame_indices = set()
            controller.token_biases = {}
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
    parser.add_argument("--e0-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    videos = load_manifest(args.manifest, "test")
    requested = (
        protocol["pilot"]["sequences"][:1]
        if args.e0_only
        else protocol["pilot"]["sequences"]
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
        "torch": torch.__version__,
        "stage": "E0" if args.e0_only else "E1",
        "scope": "gated region pilot; E2-E6 not executed",
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
        controller = UtilityController(
            UtilityHead(predictor.hidden_dim, predictor.mem_dim)
        )
        controller.enabled = False
        attach_utility_controller(predictor, controller)
        checks, total = [], 0
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
