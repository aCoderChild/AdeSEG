#!/usr/bin/env python3
"""Counterfactual regional key exclusions using native SAM2 inference and SDPA.

No new segmentation/attention architecture. Predicted memory masks define
three spatial groups; GT is used only for prompting and outcome evaluation.
Every intervention is read-only, with no current-frame memory encoding/writeback.
"""

# ruff: noqa: E402
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from evaluation.statistics import (
    holm_adjust,
    paired_wilcoxon,
    sequence_bootstrap_interval,
)
from modeling.medsam2 import build_video_predictor
from modeling.region_exclusion import RegionKeyExclusion
from modeling.region_tokens import REGION_NAMES, reference_token_regions
from scripts.memory_token_io import (
    digest,
    parameter_digest,
    synchronize_storage,
    validate_output,
)
from training.tracking import finalize_tracking, log_results_artifact, start_wandb

RANDOM_SEEDS = (20261021, 20261022, 20261023)
METRICS = ("dice", "iou", "boundary_f")


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def copy_containers(value):
    """Copy mutable state structure while sharing read-only native tensor buffers."""
    if isinstance(value, dict):
        return {key: copy_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_containers(item) for item in value)
    if isinstance(value, set):
        return value.copy()
    return value


def snapshot_before_frame(state, frame):
    """Exclude the just-written target from all output views; retain its image cache."""
    snapshot = copy_containers(state)
    for output in [
        snapshot["output_dict"],
        *snapshot.get("output_dict_per_obj", {}).values(),
    ]:
        for bucket in ("cond_frame_outputs", "non_cond_frame_outputs"):
            output[bucket] = {
                index: value for index, value in output[bucket].items() if index < frame
            }
    snapshot["frames_already_tracked"] = {
        index: value
        for index, value in snapshot["frames_already_tracked"].items()
        if index < frame
    }
    if frame not in snapshot["cached_features"]:
        raise ValueError("Target image features must already be cached")
    return snapshot


def active_bank_digest(state, frame, units, pointer_horizon):
    """Exact bytes of read spatial/pointer outputs plus the baseline target output.

    Older outputs are not cloned/re-encoded. Native next-frame logit hashes also
    verify that the original propagation bank was not changed by interventions.
    """
    wanted = {unit["frame_idx"] for unit in units} | set(
        range(max(0, frame - pointer_horizon), frame + 1)
    )
    h = hashlib.sha256()
    seen = set()

    def visit(value):
        if isinstance(value, torch.Tensor):
            key = (
                value.data_ptr(),
                tuple(value.shape),
                tuple(value.stride()),
                str(value.dtype),
            )
            if key not in seen:
                seen.add(key)
                h.update(str((value.shape, value.dtype)).encode())
                h.update(
                    value.detach()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                h.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    outputs = state["output_dict"]
    visit(
        {
            "conditioning": outputs["cond_frame_outputs"],
            "recent": {
                i: out
                for i, out in outputs["non_cond_frame_outputs"].items()
                if i in wanted
            },
            "output_frame_keys": [
                sorted(outputs[b])
                for b in ("cond_frame_outputs", "non_cond_frame_outputs")
            ],
        }
    )
    return h.hexdigest()


def evaluate(logits, truth):
    predicted = logits[0, 0].detach().cpu().numpy() > 0
    scores = calculate_scores(predicted, truth)
    return {
        "dice": scores["dice"],
        "iou": scores["iou"],
        "boundary_f": float(db_eval_boundary(truth, predicted)),
        "false_positive_on_absent": bool(predicted.any()) if not truth.any() else None,
    }


def infer_snapshot(predictor, snapshot, frame):
    """Existing author single-frame path; never add its output to any bank."""
    restored = copy_containers(snapshot)
    out, low_res = predictor.run_single_frame_inference(
        inference_state=restored,
        output_dict=restored["output_dict"],
        frame_idx=frame,
        batch_size=1,
        is_init_cond_frame=False,
        point_inputs=None,
        mask_inputs=None,
        reverse=False,
        run_mem_encoder=False,
    )
    if out["maskmem_features"] is not None or out["maskmem_pos_enc"] is not None:
        raise RuntimeError("Counterfactual unexpectedly encoded memory")
    _, logits = predictor._get_orig_video_res_output(restored, low_res)
    return logits


def sequence_statistics(rows):
    result = {}
    for region in REGION_NAMES:
        regional = [row for row in rows if row["region"] == region]
        groups = {
            "all": regional,
            "present": [r for r in regional if r["present"]],
            "absent": [r for r in regional if not r["present"]],
            "failure_present": [
                r for r in regional if r["present"] and r["baseline"]["dice"] < 0.5
            ],
            "success_present": [
                r for r in regional if r["present"] and r["baseline"]["dice"] >= 0.5
            ],
        }
        result[region] = {}
        for group, selected in groups.items():
            summary = {
                "frames": len(selected),
                "frames_with_region": sum(r["excluded_tokens"] > 0 for r in selected),
            }
            for metric in METRICS:
                if selected:
                    baseline = float(np.mean([r["baseline"][metric] for r in selected]))
                    removed = float(np.mean([r["excluded"][metric] for r in selected]))
                    random = float(
                        np.mean(
                            [
                                np.mean(
                                    [
                                        control["scores"][metric]
                                        for control in r["random"]
                                    ]
                                )
                                for r in selected
                            ]
                        )
                    )
                    summary[metric] = {
                        "baseline": baseline,
                        "excluded": removed,
                        "random": random,
                        "loss": baseline - removed,
                        "random_loss": baseline - random,
                        "excess_loss": random - removed,
                    }
                else:
                    summary[metric] = None
            if group == "absent" and selected:
                summary["false_positive_rate"] = {
                    "baseline": float(
                        np.mean(
                            [
                                r["baseline"]["false_positive_on_absent"]
                                for r in selected
                            ]
                        )
                    ),
                    "excluded": float(
                        np.mean(
                            [
                                r["excluded"]["false_positive_on_absent"]
                                for r in selected
                            ]
                        )
                    ),
                    "random": float(
                        np.mean(
                            [
                                control["scores"]["false_positive_on_absent"]
                                for r in selected
                                for control in r["random"]
                            ]
                        )
                    ),
                }
            result[region][group] = summary
    return result


def report_results(output, results):
    valid = [r for r in results if r["status"] == "completed"]
    statistics, tests = [], {}
    lines = [
        "# Regional key-exclusion counterfactuals",
        "",
        f"Completed promptable sequences: {len(valid)}.",
        "Positive loss means exclusion hurt segmentation. Excess loss = matched-random Dice minus region-excluded Dice.",
        "Primary: present-target sequence means; three fixed random seeds averaged, never best-of-N.",
        "Same cached pre-target bank, cached image features, unchanged pointers and key positions; no intervention writeback.",
        "95% CIs bootstrap whole sequences. Primary Dice tests are paired Wilcoxon, Holm across the three regions.",
        "Diagnostic one-step effects, not a deployed segmentation method or long-horizon rollout improvement.",
        "",
    ]
    for group in ("present", "failure_present", "success_present", "absent", "all"):
        lines += [
            f"## {group}",
            "",
            "| Region | Metric | Sequences | Baseline | Exclude region | Matched random | Loss | Excess loss [95% CI] |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for region in REGION_NAMES:
            for metric in METRICS:
                available = {
                    r["sequence"]: r["statistics"][region][group][metric]
                    for r in valid
                    if r["statistics"][region][group][metric] is not None
                }
                if not available:
                    continue
                row = {
                    "region": region,
                    "subset": group,
                    "metric": metric,
                    "sequences": len(available),
                }
                for key in (
                    "baseline",
                    "excluded",
                    "random",
                    "loss",
                    "random_loss",
                    "excess_loss",
                ):
                    row[key] = float(np.mean([v[key] for v in available.values()]))
                ci = (
                    sequence_bootstrap_interval(
                        {k: v["excess_loss"] for k, v in available.items()}
                    )
                    if len(available) >= 2
                    else None
                )
                row["excess_loss_ci"] = ci
                statistics.append(row)
                lines.append(
                    f"| {region} | {metric} | {len(available)} | {row['baseline']:.5f} | {row['excluded']:.5f} | {row['random']:.5f} | {row['loss']:.5f} | {row['excess_loss']:.5f} {ci if ci else 'N/A'} |"
                )
                if group == "present" and metric == "dice":
                    test = paired_wilcoxon(
                        {k: v["loss"] for k, v in available.items()},
                        {k: v["random_loss"] for k, v in available.items()},
                    )
                    tests[region] = {
                        key: None
                        if isinstance(value, float) and not np.isfinite(value)
                        else value
                        for key, value in test.items()
                    }
        lines.append("")
    if tests:
        corrected = holm_adjust({k: v["p"] for k, v in tests.items()})
        lines += [
            "## Primary paired tests",
            "",
            "| Region | Sequences | Two-sided p | Holm-adjusted p |",
            "|---|---:|---:|---:|",
        ]
        for region, test in tests.items():
            test["holm_p"] = corrected[region]
            lines.append(
                f"| {region} | {test['n']} | {test['p']:.6g} | {test['holm_p']:.6g} |"
            )
    lines += [
        "",
        "Present/absent targets and baseline failure/success subsets are reported separately.",
        "An empty region is an exact no-op, not a missing segmentation outcome. Source/patient independence is not certified.",
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return {"equal_sequence_metrics": statistics, "primary_tests": tests}


@torch.inference_mode()
def run_sequence(predictor, name, frames, audit, output, device, run):
    folder = audit / name
    reference = json.loads((folder / "summary.json").read_text())
    prompt = reference["prompt_frame"]
    if prompt is None:
        return {"sequence": name, "status": "unpromptable", "frames": 0}
    metadata = json.loads((folder / "frame_manifest.json").read_text())
    if [(f.frame_index, str(f.image_path), str(f.mask_path)) for f in frames] != [
        (f["frame"], f["image"], f["mask"]) for f in metadata
    ]:
        raise ValueError("Chronological manifest changed")
    for frame in metadata:
        if (
            digest(frame["image"]) != frame["image_sha256"]
            or digest(frame["mask"]) != frame["mask_sha256"]
        ):
            raise ValueError("Source image/mask changed")
    native = torch.load(folder / "native_memory.pt", weights_only=True)
    expected = {r["frame"]: r["prediction_sha256"] for r in native["records"]}
    del native
    labels = {}

    def label_loader(index):
        if index not in labels:
            saved = torch.load(
                folder / f"frame_{index:06d}_predicted_ratio_0.02.pt", weights_only=True
            )["labels"]
            with Image.open(folder / f"frame_{index:06d}_predicted_mask.png") as image:
                computed = reference_token_regions(
                    np.asarray(image) > 0, (64, 64), 0.02
                )
            if not torch.equal(saved, torch.from_numpy(computed["labels"])):
                raise ValueError("Predicted region assignment changed")
            labels[index] = saved
        return labels[index]

    target = output / name
    target.mkdir()
    records, seen = [], set()
    illustrations = {prompt + 1, (prompt + 1 + len(frames) - 1) // 2, len(frames) - 1}
    with tempfile.TemporaryDirectory(prefix="cached_frames_", dir=target) as temporary:
        stage_frames(frames, Path(temporary))
        state = predictor.init_state(
            video_path=temporary, offload_video_to_cpu=True, offload_state_to_cpu=False
        )
        try:
            initial = load_label_mask(frames[prompt].mask_path, [1]) == 1
            predictor.add_new_mask(
                state, frame_idx=prompt, obj_id=1, mask=torch.from_numpy(initial)
            )
            with RegionKeyExclusion(
                predictor.memory_attention, label_loader
            ) as exclusion:
                for index, _, baseline in predictor.propagate_in_video(
                    state, start_frame_idx=prompt
                ):
                    synchronize_storage(device)
                    if index in seen or tensor_hash(baseline) != expected[index]:
                        raise RuntimeError(
                            f"Baseline trajectory differs at {name}/{index}"
                        )
                    seen.add(index)
                    if index == prompt:
                        continue
                    snapshot = snapshot_before_frame(state, index)
                    native_units = predictor.memory_attention.last_trace["units"]
                    before = active_bank_digest(
                        state, index, native_units, predictor.max_obj_ptrs_in_encoder
                    )
                    truth = load_label_mask(frames[index].mask_path, [1]) == 1
                    base_scores = evaluate(baseline, truth)
                    pending = []

                    def infer(region, seed=None):
                        exclusion.condition(region, index, name, seed)
                        try:
                            prediction = infer_snapshot(predictor, snapshot, index)
                            synchronize_storage(device)
                        finally:
                            exclusion.enabled = False
                        traces = exclusion.rows.copy()
                        if len(traces) != len(predictor.memory_attention.layers):
                            raise RuntimeError(
                                "Exclusion did not cover every memory-attention layer"
                            )
                        if any(
                            t["mask_sha256"] != traces[0]["mask_sha256"] for t in traces
                        ):
                            raise RuntimeError(
                                "Different exclusion masks across layers"
                            )
                        return prediction, traces[0]

                    no_op, layout = infer(None)
                    if not torch.equal(no_op, baseline):
                        raise RuntimeError(
                            f"Restored cached state/no-op differs at {name}/{index}"
                        )
                    if layout["memory_units"] != [
                        {
                            key: unit[key]
                            for key in ("start", "end", "frame_idx", "conditioning")
                        }
                        for unit in native_units
                    ]:
                        raise RuntimeError(
                            "Restoring state changed baseline memory-key order"
                        )
                    del no_op
                    if index in illustrations:
                        Image.fromarray(
                            (baseline[0, 0].cpu().numpy() > 0).astype(np.uint8) * 255
                        ).save(target / f"frame_{index:06d}_baseline.png")
                    for region_index, region in enumerate(REGION_NAMES):
                        prediction, trace = infer(region_index)
                        if (
                            trace["memory_units"] != layout["memory_units"]
                            or trace["pointer_tokens"] != layout["pointer_tokens"]
                        ):
                            raise RuntimeError(
                                "Intervention changed the native bank layout"
                            )
                        row = {
                            "sequence": name,
                            "frame": index,
                            "region": region,
                            "present": bool(truth.any()),
                            "baseline": base_scores,
                            "excluded": evaluate(prediction, truth),
                            "excluded_prediction_sha256": tensor_hash(prediction),
                            **trace,
                            "random": [],
                        }
                        if index in illustrations:
                            Image.fromarray(
                                (prediction[0, 0].cpu().numpy() > 0).astype(np.uint8)
                                * 255
                            ).save(target / f"frame_{index:06d}_exclude_{region}.png")
                        del prediction
                        for seed in RANDOM_SEEDS:
                            prediction, random_trace = infer(region_index, seed)
                            if (
                                random_trace["excluded_per_memory_frame"]
                                != trace["excluded_per_memory_frame"]
                                or random_trace["pointer_tokens"]
                                != trace["pointer_tokens"]
                                or random_trace["memory_units"]
                                != layout["memory_units"]
                            ):
                                raise RuntimeError(
                                    "Random null is not count/temporal/pointer matched"
                                )
                            row["random"].append(
                                {
                                    "seed": seed,
                                    "scores": evaluate(prediction, truth),
                                    "prediction_sha256": tensor_hash(prediction),
                                    **random_trace,
                                }
                            )
                            del prediction
                        pending.append(row)
                        (output / "progress.json").write_text(
                            json.dumps(
                                {
                                    "state": "running",
                                    "sequence": name,
                                    "frame": index,
                                    "total_frames": len(frames),
                                    "last_region": region,
                                }
                            )
                            + "\n"
                        )
                        print(
                            f"[exclude] {name} frame={index}/{len(frames) - 1} region={region} keys={trace['excluded_tokens']} random=3",
                            flush=True,
                        )
                    restored, _ = infer(None)
                    if not torch.equal(restored, baseline):
                        raise RuntimeError(
                            "Post-intervention restored prediction differs"
                        )
                    del restored
                    after = active_bank_digest(
                        state, index, native_units, predictor.max_obj_ptrs_in_encoder
                    )
                    if after != before:
                        raise RuntimeError("Native read-memory/target buffers changed")
                    with (target / "frames.jsonl").open("a") as stream:
                        for row in pending:
                            row.update(
                                cached_restore_exact=True,
                                active_bank_unchanged=True,
                                counterfactual_memory_encoding=False,
                                counterfactual_writeback=False,
                            )
                            stream.write(json.dumps(row) + "\n")
                            records.append(row)
                    run.log(
                        {
                            "progress/frame": index,
                            "progress/sequence": int(name[3:]),
                            "progress/validated_cases": len(records),
                        }
                    )
        finally:
            predictor.reset_state(state)
    if seen != set(range(prompt, len(frames))):
        raise RuntimeError("Incomplete exclusion target coverage")
    result = {
        "sequence": name,
        "status": "completed",
        "frames": len(frames) - prompt - 1,
        "native_trajectory_exact": True,
        "cached_restore_exact": True,
        "active_bank_unchanged": True,
        "statistics": sequence_statistics(records),
    }
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    log_results_artifact(
        run,
        output,
        [str(p.relative_to(output)) for p in target.iterdir() if p.is_file()],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl"
    )
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument(
        "--sequence",
        action="append",
        help="Default all 23 IDs, with seq3 first as correctness gate",
    )
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocols/memory_token_regions.json").read_text())
    if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise ValueError("Unregistered checkpoint")
    if not json.loads((args.audit_dir / "delivery.json").read_text())["completed"]:
        raise ValueError("Reference assignment audit is incomplete")
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for key, name in (
        ("WANDB_CACHE_DIR", "wandb_cache"),
        ("WANDB_DATA_DIR", "wandb_data"),
    ):
        (output / name).mkdir()
        os.environ[key] = str(output / name)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        random_seeds=RANDOM_SEEDS,
        primary_label_source="predicted mask, width 0.02",
        operator="Native SDPA additive -inf mask, all 4 memory cross-attention layers",
        pointers="unchanged",
        random_null="Uniform sampling matched separately within each memory frame; mean of three seeds, never best-of-N",
        effect="one-target counterfactual; no intervention memory encoding/writeback",
        checkpoint_sha256=digest(args.checkpoint),
        started_at_unix=time.time(),
    )
    (output / "setup.json").write_text(json.dumps(config, indent=2) + "\n")
    sources = [
        "scripts/run_region_exclusions.py",
        "scripts/memory_token_io.py",
        "modeling/region_exclusion.py",
        "modeling/region_attention.py",
        "modeling/region_tokens.py",
        "training/tracking.py",
        "protocols/memory_token_regions.json",
        "external/MedSAM2/sam2/sam2_video_predictor.py",
        "external/MedSAM2/sam2/modeling/sam/transformer.py",
        "external/MedSAM2/sam2/modeling/sam2_base.py",
        "evaluation/davis2017/metrics.py",
        "evaluation/davis2017/LICENSE",
    ]
    for relative in sources:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        "06 | Regional causal exclusions | 3 matched random controls",
        "online",
        "https://forge.coreweave.com/api/wandb",
        config,
        output,
        ["region-exclusion", "counterfactual", "prediction-defined"],
    )
    try:
        videos = {
            **load_manifest(args.manifest, "dev"),
            **load_manifest(args.manifest, "test"),
        }
        if set(videos) != {f"seq{i}" for i in range(1, 24)}:
            raise ValueError("Expected all 23 source sequences")
        names = args.sequence or ["seq3"] + [f"seq{i}" for i in range(1, 24) if i != 3]
        if len(names) != len(set(names)) or any(name not in videos for name in names):
            raise ValueError("Invalid sequence list")
        predictor = build_video_predictor(
            "configs/sam2.1_hiera_b+1024.yaml", args.checkpoint, args.device
        ).eval()
        predictor.requires_grad_(False)
        before = parameter_digest(predictor)
        results = []
        for name in names:
            results.append(
                run_sequence(
                    predictor,
                    name,
                    videos[name],
                    args.audit_dir,
                    output,
                    args.device,
                    run,
                )
            )
            if parameter_digest(predictor) != before:
                raise RuntimeError("Frozen model parameters changed")
            (output / "sequences.json").write_text(json.dumps(results, indent=2) + "\n")
            report_results(output, results)
        summary = {
            "completed": True,
            "completed_sequences": sum(r["status"] == "completed" for r in results),
            "target_frames": sum(r["frames"] for r in results),
            "sequences": results,
            **report_results(output, results),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        finalize_tracking(
            run,
            output,
            {
                "completed": True,
                "target_frames": summary["target_frames"],
                "completed_sequences": summary["completed_sequences"],
            },
            ["setup.json", "summary.json", "results.md"]
            + [
                str(p.relative_to(output))
                for p in (output / "source").rglob("*")
                if p.is_file()
            ],
        )
    except Exception as error:
        (output / "error.json").write_text(
            json.dumps({"completed": False, "error": str(error)}) + "\n"
        )
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
