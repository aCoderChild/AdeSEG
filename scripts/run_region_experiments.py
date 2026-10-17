#!/usr/bin/env python3
"""Sequence shards for E2/E3 and E6, using the verified E1 replay and SAM2 loop.

E4/E5 consume these rows with analyze_region_experiments.py. No stage in this
entry point declares the complete study finished. Run on Kaggle; outputs must
be an explicit Kaggle or mounted Google Drive path.
"""
# ruff: noqa: E402

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "external/MedSAM2")]
from datasets.io import load_label_mask
from datasets.video import foreground_labels, load_manifest
from evaluation.segmentation import calculate_scores
from evaluation.utility_retrieval import temporal_summary
from modeling.medsam2 import build_video_predictor
from modeling.region_tokens import (
    CATEGORIES,
    indices_to_masks,
    indices_to_value_replacements,
    matched_experiments,
    reliability_bias,
    token_categories,
)
from modeling.utility_memory import (
    UtilityController,
    UtilityHead,
    attach_utility_controller,
)
from scripts.audit_region_initialization import prompt_index
from scripts.infer import stage_frames
from scripts.generate_utility_labels import replace_spatial_memory
from scripts.run_region_token_pilot import (
    metrics,
    run_sequence,
    swap_replacements,
    write_json,
)
from training.tracking import log_results_artifact, start_wandb


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require_gate(directory):
    """Require independently recorded exactness and the complete E1 decision."""
    directory = Path(directory)
    sanity_files = list(directory.rglob("sanity.jsonl"))
    decisions = list(directory.rglob("decision_verified.json")) or list(
        directory.rglob("decision.json")
    )
    if not sanity_files or not decisions:
        raise ValueError(
            "Gate directory must contain E0 sanity.jsonl and the E1 decision"
        )
    checks = [
        json.loads(line)
        for path in sanity_files
        for line in path.read_text().splitlines()
        if line
    ]
    required = (
        "detached_noop_exact",
        "empty_token_mask_exact",
        "zero_bias_exact",
        "independent_restore_exact",
        "mutation_changed_prediction",
        "full_token_equals_frame",
    )
    if not checks or any(
        not c.get("passed")
        or any(c.get(key) is not True for key in required)
        or c.get("attention_reconstruction_max_abs", float("inf")) >= 1e-5
        for c in checks
    ):
        raise ValueError("E0 exactness gate failed or is incomplete")
    decision = json.loads(decisions[0].read_text())
    if not (
        decision.get("stage") == "E1"
        and decision.get("all_required_sequences") is True
        and decision.get("gate_pass") is True
        and decision.get("decision") == "proceed_to_E2"
    ):
        raise ValueError("E1 scientific gate has not passed")
    return {
        "decision_sha256": digest(decisions[0]),
        "sanity_files": [digest(p) for p in sorted(sanity_files)],
    }


def validate_output(path):
    path = Path(path).resolve()
    allowed = (
        path.is_relative_to(Path("/kaggle/working"))
        or path.is_relative_to(
            Path(
                "/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs"
            )
        )
        or path.is_relative_to(Path("/content/drive/MyDrive/AdeSEG/outputs"))
    )
    if not allowed:
        raise ValueError(
            "Experiment outputs must be on Kaggle or the requested Google Drive"
        )
    return path


def donor_inventory(directory):
    inventory = []
    for path in sorted(Path(directory).glob("donor_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if data["values"].shape != (4096, 64) or data["categories"].shape != (
                64,
                64,
            ):
                raise ValueError(f"Invalid donor shape: {path}")
            inventory.append(
                {
                    "path": str(path),
                    "video": str(data["video"]),
                    "relative_time": float(data["relative_time"]),
                    "label_id": int(data["label_id"]),
                }
            )
    if not inventory:
        raise ValueError("Swap requires a completed dev donor calibration")
    return inventory


def validate_method_split(dataset, test, sequences, sources, raw):
    polyp = json.loads((ROOT / "datasets/splits/polypgen/protocol_a.json").read_text())
    cholec = json.loads(
        (ROOT / "protocols/region_tokens_v1_2_cholec_amendment.json").read_text()
    )["cholecseg8k"]["method_split"]
    allowed = (
        {f"seq{n}" for n in polyp["test" if test else "dev"]}
        if dataset == "polypgen"
        else set(cholec["test_source_videos" if test else "dev_source_videos"])
        if dataset == "cholecseg8k"
        else {
            r["video_id"]
            for r in raw
            if r["split"] in ({"val"} if test else {"train", "holdout"})
        }
    )
    if any(sources[v] not in allowed for v in sequences):
        raise ValueError(
            "Method/calibration shard crosses the registered dev/test split"
        )


def lean_anchor_similarity(memory, units, taxonomy):
    """Reuse Lean-SAM2's three pure similarity methods verbatim.

    Loading just these methods avoids importing a second incompatible `sam2`
    package into the instrumented predictor process. Anchor support here is all
    prompted foreground spatial tokens; this diagnostic is explicitly distinct
    from the complete upstream TAMP baseline and its usage-selected anchors.
    """
    path = ROOT / "external/Lean-SAM2/sam2/modeling/sam2_base.py"
    names = {
        "_flatten_similarity_tokens",
        "_compute_similarity_maxcos",
        "_compute_anchor_similarity_maxcos",
    }
    tree = ast.parse(path.read_text())
    source_class = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SAM2Base"
    )
    methods = [
        n
        for n in source_class.body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if len(methods) != len(names):
        raise RuntimeError("Upstream Lean-SAM2 similarity API changed")
    namespace = {"torch": torch, "F": F}
    for method in methods:
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                str(path),
                "exec",
            ),
            namespace,
        )
    instance = SimpleNamespace()
    for name in names:
        setattr(instance, name, namespace[name].__get__(instance))
    anchors = [
        memory[0, u["start"] : u["end"]][
            np.isin(taxonomy[u["frame_idx"]].ravel(), [0, 1])
        ]
        for u in units
        if u["conditioning"]
    ]
    if not anchors or not sum(len(a) for a in anchors):
        return None
    anchors = torch.cat(anchors)
    result = {
        "source_sha256": digest(path),
        "criterion": "Lean-maxcos/all-prompt-foreground-anchors",
    }
    for name, conditioning in (("conditioning", True), ("recent", False)):
        scores = {category: [] for category in CATEGORIES}
        for unit in units:
            if bool(unit["conditioning"]) != conditioning:
                continue
            values = memory[0, unit["start"] : unit["end"]]
            # Bounded chunks avoid a 24k x 4k temporary similarity tensor.
            similarities = torch.cat(
                [
                    instance._compute_anchor_similarity_maxcos(chunk, anchors)
                    for chunk in values.split(256)
                ]
            )
            labels = taxonomy[unit["frame_idx"]].ravel()
            for i, category in enumerate(CATEGORIES):
                scores[category].extend(similarities[labels == i].tolist())
        result[name] = {
            category: float(np.mean(v)) if v else None for category, v in scores.items()
        }
    return result


def clear_controller(controller):
    controller.excluded_token_masks = {}
    controller.excluded_frame_indices = set()
    controller.token_biases = {}
    controller.token_value_replacements = {}
    controller.token_memory_replacements = {}
    controller.token_mask_policy = None
    controller.token_bias_policy = None
    controller.token_value_replacement_policy = None
    controller.token_memory_replacement_policy = None
    controller.exclude_pointers = False


@torch.inference_mode()
def rollout(predictor, controller, frames, protocol, config, output, run):
    """Native persistent propagation; each arm owns an independent state.

    The unmodified arm defines failure episodes once. Predictions, never GT,
    enter persistent memory. Analysis categories may be oracle; method
    categories always use the stored predictions, except the labeled oracle
    ablation. Image features are shared across arms by a sequence-local cache.
    """
    label_id, label_ids = config["label_id"], config["label_ids"]
    prompt = prompt_index(frames, protocol, label_id, label_ids)

    def gt(t):
        return load_label_mask(frames[t].mask_path, label_ids) == label_id

    results = []
    feature_cache = tempfile.TemporaryDirectory(
        prefix="region_feature_cache_", dir=output
    )
    original_get_features = predictor._get_image_feature

    def cached_features(state, frame_idx, batch_size):
        path = Path(feature_cache.name) / f"{frame_idx}.pt"
        if frame_idx not in state["cached_features"] and path.exists():
            state["cached_features"] = {
                frame_idx: torch.load(
                    path, weights_only=True, map_location=predictor.device
                )
            }
        result = original_get_features(state, frame_idx, batch_size)
        if not path.exists():
            torch.save(state["cached_features"][frame_idx], path)
        return result

    predictor._get_image_feature = cached_features
    is_method = config["stage"].startswith("E6")
    arms = [("baseline", None, 0)]
    if is_method:
        names = (
            (
                "full",
                "no_reliability",
                "boundary_only",
                "artifact_off",
                "random",
                "oracle",
            )
            if config["stage"] == "E6_test"
            else ("full",)
        )
        arms += [(name, None, config.get("seed", 0)) for name in names]
        if config["stage"] == "E6_test":
            arms.append(("efficient_saliency", None, 0))
    else:
        for bank in ("conditioning", "recent"):
            for category in CATEGORIES:
                arms.append((category, bank, 0))
            for seed in protocol["budget"]["seeds"]:
                arms.extend([("random", bank, seed), ("spatial_random", bank, seed)])
        arms += [("all_spatial", None, 0), ("pointers_only", None, 0)]
    with tempfile.TemporaryDirectory(prefix="region_persistent_") as directory:
        stage_frames(frames, Path(directory))
        try:
            for category, bank, seed in arms:
                clear_controller(controller)
                state = predictor.init_state(
                    video_path=directory,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )
                predictor.add_new_mask(
                    state, frame_idx=prompt, obj_id=1, mask=torch.from_numpy(gt(prompt))
                )
                predicted = {prompt: gt(prompt)}
                maps, decisions = {}, {}
                original_single_frame = predictor._run_single_frame_inference
                previous_prediction, previous_target = None, None
                arm_rows = []

                def taxonomy(t):
                    if t not in maps:
                        mask = (
                            gt(t)
                            if config.get("category_source", "oracle") == "oracle"
                            or category == "oracle"
                            else predicted[t]
                        )
                        image = cv2.cvtColor(
                            cv2.imread(str(frames[t].image_path)), cv2.COLOR_BGR2RGB
                        )
                        maps[t] = token_categories(
                            mask,
                            band=config.get("band", 1),
                            thresholds=config.get("thresholds", [0.1, 0.9]),
                            image=image,
                        )
                    return maps[t]

                def selection(units):
                    t = max(u["frame_idx"] + u["temporal_distance"] for u in units)
                    if t < prompt + 7 or category == "baseline":
                        return [], np.array([], dtype=int), np.array([], dtype=int)
                    chosen = [
                        u
                        for u in units
                        if bank is None
                        or bool(u["conditioning"]) == (bank == "conditioning")
                    ]
                    labels = np.concatenate(
                        [taxonomy(u["frame_idx"])["categories"].ravel() for u in chosen]
                    )
                    k, _, experiments = matched_experiments(
                        labels.reshape(-1, 64, 64), [seed]
                    )
                    selected = next(
                        (
                            s
                            for name, s_seed, s in experiments
                            if name == category and s_seed == seed
                        ),
                        None,
                    )
                    if category == "all_spatial":
                        selected = np.arange(len(labels))
                    decisions[t] = {
                        "k": k,
                        "applied_tokens": len(selected) if selected is not None else 0,
                        "feasible": selected is not None,
                    }
                    return chosen, labels, selected

                def mask_policy(units):
                    chosen, _, selected = selection(units)
                    return (
                        indices_to_masks(chosen, selected)
                        if chosen and selected is not None
                        else {}
                    )

                def value_policy(memory, units):
                    chosen, labels, selected = selection(units)
                    if not chosen or selected is None:
                        return {}
                    if config["intervention"] == "mean_substitution":
                        means = np.asarray(
                            [config["means"][CATEGORIES[c]] for c in labels[selected]]
                        )
                        return indices_to_value_replacements(chosen, selected, means)
                    t = max(u["frame_idx"] + u["temporal_distance"] for u in chosen)
                    replacement, donor_videos = swap_replacements(
                        chosen,
                        selected,
                        labels,
                        config["donors"],
                        config["video"],
                        t / max(1, len(frames) - 1),
                    )
                    decisions[t].update(
                        donor_videos=donor_videos,
                        donor_position_distance=0.0 if replacement else None,
                        feasible=replacement is not None,
                        applied_tokens=len(selected) if replacement else 0,
                    )
                    return replacement or {}

                def bias_policy(units):
                    t = max(u["frame_idx"] + u["temporal_distance"] for u in units)
                    if t < prompt + 7:
                        return {}
                    return reliability_bias(
                        units,
                        {u["frame_idx"]: taxonomy(u["frame_idx"]) for u in units},
                        config["weights"],
                        config["scale"],
                        "full" if category == "oracle" else category,
                        seed,
                    )

                def saliency_read(**kwargs):
                    """Measure this target's saliency, then rerun without writing the measurement."""
                    t = kwargs["frame_idx"]
                    controller.excluded_token_masks = {}
                    if t < prompt + 7:
                        return original_single_frame(**kwargs)
                    predictor.memory_attention.capture_attention = True
                    try:
                        original_single_frame(**{**kwargs, "run_mem_encoder": False})
                        trace = predictor.memory_attention.last_trace
                        units = trace["units"]
                        labels = np.concatenate(
                            [
                                taxonomy(u["frame_idx"])["categories"].ravel()
                                for u in units
                            ]
                        )
                        k = int((labels == 0).sum())
                        scores = np.concatenate(
                            [
                                torch.stack(
                                    [
                                        layer[..., u["start"] : u["end"]]
                                        .mean((0, 1, 2))
                                        .cpu()
                                        for layer in trace["attention"]
                                    ]
                                )
                                .mean(0)
                                .numpy()
                                for u in units
                            ]
                        )
                        selected = np.argsort(scores, kind="stable")[:k]
                        controller.excluded_token_masks = indices_to_masks(
                            units, selected
                        )
                        decisions[t] = {"feasible": True, "k": k, "applied_tokens": k}
                    finally:
                        predictor.memory_attention.capture_attention = False
                        if predictor.memory_attention.last_trace:
                            predictor.memory_attention.last_trace["attention"] = []
                        for layer in predictor.memory_attention.layers:
                            layer.cross_attn_image.last_attention = None
                    return original_single_frame(**kwargs)

                if category == "pointers_only":

                    def pointer_policy(units):
                        controller.exclude_pointers = (
                            max(u["frame_idx"] + u["temporal_distance"] for u in units)
                            >= prompt + 7
                        )
                        return {}

                    controller.token_mask_policy = pointer_policy
                elif category == "efficient_saliency":
                    predictor._run_single_frame_inference = saliency_read
                elif category != "baseline" and is_method:
                    controller.token_bias_policy = bias_policy
                elif category != "baseline":
                    if (
                        config["intervention"] == "attention_mask"
                        or category == "all_spatial"
                    ):
                        controller.token_mask_policy = mask_policy
                    elif config["intervention"] == "mean_substitution":
                        controller.token_value_replacement_policy = value_policy
                    else:
                        controller.token_memory_replacement_policy = value_policy
                started = time.perf_counter()
                try:
                    for t, _, logits in predictor.propagate_in_video(
                        state, start_frame_idx=prompt
                    ):
                        target = gt(t)
                        prediction = logits[0, 0].float().cpu().numpy() > 0
                        predicted[t] = prediction
                        if not is_method and t < prompt + 7:
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
                        if t > prompt:
                            void = np.isin(
                                load_label_mask(frames[t].mask_path, label_ids),
                                config.get("ignore_labels", []),
                            )
                            stored = state["output_dict"]["non_cond_frame_outputs"][t][
                                "pred_masks"
                            ]
                            scores = metrics(
                                logits[:, :1],
                                target,
                                void,
                                decoder_logits=stored[:, :1],
                            )
                            stability = (
                                None
                                if previous_prediction is None
                                else abs(
                                    calculate_scores(prediction, previous_prediction)[
                                        "iou"
                                    ]
                                    - calculate_scores(target, previous_target)["iou"]
                                )
                            )
                            row = {
                                "video": config["video"],
                                "frame": int(t),
                                "category": category,
                                "bank": bank,
                                "seed": seed,
                                "mode": "persistent",
                                "dataset": config["dataset"],
                                "label_id": label_id,
                                "state": "skipped" if void.all() else "complete",
                                "skip_reason": "fully_unannotated_target"
                                if void.all()
                                else None,
                                "intervention_kind": (
                                    "none"
                                    if category == "baseline"
                                    else "attention_mask"
                                    if category
                                    in {
                                        "all_spatial",
                                        "pointers_only",
                                        "efficient_saliency",
                                    }
                                    else config["intervention"]
                                ),
                                "category_source": "oracle"
                                if category == "oracle"
                                else config.get("category_source", "oracle"),
                                "metrics": scores,
                                "stability_error": stability,
                                "logits_sha256": hashlib.sha256(
                                    logits.detach().float().cpu().numpy().tobytes()
                                ).hexdigest(),
                                **decisions.get(t, {}),
                            }
                            row.update(
                                source_video=config.get(
                                    "source_video", config["video"]
                                ),
                                split=config["split"],
                            )
                            row.update(
                                memory_source="natural"
                                if is_method
                                else "c0_initialized_then_predicted",
                                band=config.get("band", 1),
                                thresholds=config.get("thresholds", [0.1, 0.9]),
                            )
                            arm_rows.append(row)
                            with (output / "rollout.jsonl").open("a") as stream:
                                stream.write(json.dumps(row, allow_nan=False) + "\n")
                        previous_prediction, previous_target = prediction, target
                    elapsed = time.perf_counter() - started
                    summary = {
                        "category": category,
                        "bank": bank,
                        "seed": seed,
                        "frames": len(arm_rows),
                        "fps": len(arm_rows) / elapsed,
                        "fps_scope": "end-to-end propagation, metric computation and feature-cache IO",
                        **temporal_summary([r["metrics"]["dice"] for r in arm_rows]),
                    }
                    results.append(summary)
                    run.log(
                        {
                            "progress/arms": len(results),
                            "progress/frames": len(arm_rows),
                            "cost/fps": summary["fps"],
                        }
                    )
                    log_results_artifact(run, output, ["rollout.jsonl"])
                finally:
                    predictor.memory_attention.capture_attention = False
                    predictor._run_single_frame_inference = original_single_frame
                    clear_controller(controller)
                    predictor.reset_state(state)
        finally:
            predictor._get_image_feature = original_get_features
            feature_cache.cleanup()
    write_json(output / "temporal_summary.json", {"arms": results})
    if is_method and config["scale"] == 0:
        baseline = {}
        for line in (output / "rollout.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["category"] == "baseline":
                baseline[row["frame"]] = row["logits_sha256"]
            elif (
                row["category"] != "efficient_saliency"
                and baseline.get(row["frame"]) != row["logits_sha256"]
            ):
                raise RuntimeError(
                    "E6 lambda=0 failed the independent rollout no-op check"
                )
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["E2", "E3", "E2_calibration", "E6_dev", "E6_test"],
        required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--dataset", choices=["polypgen", "cholecseg8k", "davis"], required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "protocols/region_tokens_v1_1.json"
    )
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--label-id", action="append", type=int)
    parser.add_argument(
        "--mode", choices=["single_step", "persistent"], default="single_step"
    )
    parser.add_argument(
        "--intervention",
        choices=["attention_mask", "mean_substitution", "swap"],
        default="attention_mask",
    )
    parser.add_argument(
        "--memory-condition", choices=["c0", "c1", "c2", "natural"], default="c0"
    )
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--method-config", type=Path)
    parser.add_argument("--band", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--thresholds", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    gate = require_gate(args.gate_dir)
    output = validate_output(args.output_dir)
    protocol = json.loads(args.protocol.read_text())
    videos = load_manifest(args.manifest, args.split)
    if (
        len(args.sequence) != len(set(args.sequence))
        or set(args.sequence) - videos.keys()
    ):
        raise ValueError("Sequence shard contains duplicates or missing videos")
    if args.thresholds not in [[0.05, 0.95], [0.1, 0.9], [0.2, 0.8]]:
        raise ValueError(
            "Coverage thresholds are outside the registered sensitivity grid"
        )
    if args.stage == "E3" and (
        args.mode != "single_step" or args.intervention != "attention_mask"
    ):
        raise ValueError("E3 is single-step attention masking")
    if args.stage == "E2" and args.memory_condition != "c0":
        raise ValueError("Corrupted and natural single-step conditions belong to E3")
    if args.stage == "E2_calibration" and (
        args.mode != "single_step"
        or args.intervention != "attention_mask"
        or args.memory_condition != "c0"
        or args.band != 1
        or args.thresholds != [0.1, 0.9]
    ):
        raise ValueError("Calibration requires the c0 single-step primary taxonomy")
    if args.stage.startswith("E6") and args.intervention != "attention_mask":
        raise ValueError("E6 uses the frozen region bias, not substitution operators")
    raw = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    sources = {r["video_id"]: r.get("source_video", r["video_id"]) for r in raw}
    if args.stage in {"E2_calibration", "E6_dev", "E6_test"}:
        validate_method_split(
            args.dataset, args.stage == "E6_test", args.sequence, sources, raw
        )
    config = {
        "stage": args.stage,
        "dataset": args.dataset,
        "mode": args.mode,
        "memory_condition": args.memory_condition,
        "intervention": args.intervention,
        "band": args.band,
        "thresholds": args.thresholds,
        "anchor_similarity": args.stage != "E2_calibration",
        "category_source": "deployable" if args.stage.startswith("E6") else "oracle",
        "ignore_labels": [13] if args.dataset == "cholecseg8k" else [],
        "save_donors": args.stage == "E2_calibration",
    }
    if args.stage.startswith("E6"):
        if not args.method_config:
            raise ValueError("E6 requires a frozen dev-derived method config")
        method = json.loads(args.method_config.read_text())
        if method.get("selection_split") != "dev" or method.get("frozen") is not True:
            raise ValueError("E6 method config is not frozen on dev")
        if args.stage == "E6_test" and method.get("evaluation_ready") is not True:
            raise ValueError(
                "Freeze lambda from the completed dev comparison before E6 test"
            )
        if method.get("checkpoint_sha256") != digest(args.checkpoint):
            raise ValueError(
                "Method was calibrated for a different or unrecorded checkpoint"
            )
        config.update(
            weights=method["weights"],
            scale=method["scale"],
            method_sha256=digest(args.method_config),
            memory_condition="natural",
            category_source="deployable",
            mode="persistent",
            intervention="region_bias",
        )
    if args.intervention != "attention_mask":
        if not args.calibration_dir:
            raise ValueError("Mean/swap requires dev calibration")
        provenance = json.loads(
            (args.calibration_dir / "calibration_provenance.json").read_text()
        )
        if (
            provenance.get("dataset") != args.dataset
            or provenance.get("selection_split") != "dev"
        ):
            raise ValueError("Calibration dataset/split mismatch")
        calibration_status = json.loads(
            (args.calibration_dir / "status.json").read_text()
        )
        calibration_setup = json.loads(
            (args.calibration_dir / "setup.json").read_text()
        )
        if (
            calibration_status.get("state") != "complete"
            or calibration_status.get("stage") != "E2_calibration"
            or calibration_setup.get("checkpoint_sha256") != digest(args.checkpoint)
            or calibration_setup.get("protocol_sha256") != digest(args.protocol)
        ):
            raise ValueError(
                "Calibration is incomplete or uses another checkpoint/protocol"
            )
        config["calibration_sha256"] = digest(
            args.calibration_dir / "calibration_provenance.json"
        )
        if args.intervention == "mean_substitution":
            calibration = json.loads(
                (args.calibration_dir / "value_calibration.json").read_text()
            )
            config["means"] = {
                k: v["mean"] for k, v in calibration["categories"].items()
            }
            if set(config["means"]) != set(CATEGORIES) or any(
                np.asarray(v).shape != (64,) or not np.isfinite(v).all()
                for v in config["means"].values()
            ):
                raise ValueError("Invalid category means")
        else:
            config["donors"] = donor_inventory(args.calibration_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    setup = {
        **config,
        "gate": gate,
        "manifest_sha256": digest(args.manifest),
        "checkpoint_sha256": digest(args.checkpoint),
        "protocol_sha256": digest(args.protocol),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "execution_protocol_sha256": digest(
            ROOT / "protocols/region_tokens_execution_v1.json"
        ),
        "sequences": args.sequence,
        "split": args.split,
        "labels": args.label_id,
        "full_study_complete": False,
    }
    if (output / "setup.json").exists() and json.loads(
        (output / "setup.json").read_text()
    ) != setup:
        raise ValueError("Resume configuration differs; use a new shard directory")
    write_json(output / "setup.json", setup)
    write_json(
        output / "execution_protocol.json",
        json.loads((ROOT / "protocols/region_tokens_execution_v1.json").read_text()),
    )
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        f"region-tokens-{args.stage}",
        "online",
        "https://forge.coreweave.com/api/wandb",
        setup,
        output,
        ["region-tokens-v1", args.stage],
    )
    started = time.time()
    calibration = (
        {name: {"sum": np.zeros(64, np.float64), "count": 0} for name in CATEGORIES}
        if args.stage == "E2_calibration"
        else None
    )
    try:
        predictor = build_video_predictor(
            protocol["model"]["config"], args.checkpoint, args.device
        ).eval()
        controller = UtilityController(
            UtilityHead(predictor.hidden_dim, predictor.mem_dim)
        )
        controller.enabled = False
        attach_utility_controller(predictor, controller)
        completed = []
        for video in args.sequence:
            frames = videos[video]
            labels = foreground_labels(frames)
            objects = args.label_id or [
                n for n in labels if n not in config["ignore_labels"]
            ]
            with (output / "eligibility.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "video": video,
                            "source_video": sources[video],
                            "objects": objects,
                            "state": "eligible" if objects else "skipped",
                            "reason": None if objects else "no_labeled_foreground",
                        }
                    )
                    + "\n"
                )
            for label_id in objects:
                unit = output / f"{video}__object{label_id}"
                if unit.exists():
                    status = unit / "status.json"
                    if (
                        not status.is_file()
                        or json.loads(status.read_text()).get("state") != "complete"
                        or calibration is not None
                    ):
                        raise ValueError(
                            f"Partial/calibration shard exists; preserve it and use a new output: {unit}"
                        )
                    completed.append(unit.name)
                    continue
                unit.mkdir()
                settings = {
                    **config,
                    "video": video,
                    "source_video": sources[video],
                    "split": args.split,
                    "label_id": label_id,
                    "label_ids": labels,
                }
                if args.dataset == "cholecseg8k" and "donors" in settings:
                    settings["donors"] = [
                        d for d in settings["donors"] if d["label_id"] == label_id
                    ]
                try:
                    prompt_index(frames, protocol, label_id, labels)
                    if (
                        len(frames)
                        <= prompt_index(frames, protocol, label_id, labels) + 7
                    ):
                        raise ValueError("Too short for a filled bank")
                except ValueError as exc:
                    write_json(
                        unit / "status.json",
                        {"state": "skipped", "reason": str(exc), "label_id": label_id},
                    )
                    continue
                print(
                    f"[region-study] {args.stage} {video} object={label_id} starting",
                    flush=True,
                )
                if args.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                if settings["mode"] == "persistent":
                    sanity_output = unit / "E0"
                    sanity_output.mkdir()
                    checks, _ = run_sequence(
                        predictor,
                        controller,
                        video,
                        frames,
                        protocol,
                        sanity_output,
                        run,
                        True,
                        experiment=settings,
                    )
                    if not checks or not all(c["passed"] for c in checks):
                        raise RuntimeError(
                            "Fresh E0 failed before persistent propagation"
                        )
                    arms = rollout(
                        predictor, controller, frames, protocol, settings, unit, run
                    )
                    with (unit / "rollout.jsonl").open() as stream:
                        rows = sum(1 for _ in stream)
                else:
                    checks, rows = run_sequence(
                        predictor,
                        controller,
                        video,
                        frames,
                        protocol,
                        unit,
                        run,
                        False,
                        value_calibration=calibration,
                        mean_values=settings.get("means"),
                        targets_per_sequence=protocol["pilot"]["targets_per_sequence"]
                        if calibration is not None
                        else None,
                        experiment=settings,
                    )
                write_json(
                    unit / "status.json",
                    {
                        "state": "complete",
                        "stage": args.stage,
                        "rows": rows,
                        "arms": arms if settings["mode"] == "persistent" else None,
                        "peak_gpu_scope": "stage peak including fresh E0 instrumentation",
                        "fresh_e0_pass": bool(checks)
                        and all(c["passed"] for c in checks),
                    },
                )
                completed.append(unit.name)
                run.log(
                    {
                        "progress/completed_units": len(completed),
                        "cost/gpu_hours": (time.time() - started) / 3600,
                        "cost/peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20
                        if args.device.startswith("cuda")
                        else 0,
                    }
                )
                log_results_artifact(
                    run,
                    output,
                    [str(p.relative_to(output)) for p in unit.glob("*") if p.is_file()],
                )
        if calibration is not None:
            if any(not c["count"] for c in calibration.values()):
                raise ValueError("Insufficient dev calibration category coverage")
            write_json(
                output / "value_calibration.json",
                {
                    "categories": {
                        name: {
                            "count": item["count"],
                            "mean": (item["sum"] / item["count"]).tolist(),
                        }
                        for name, item in calibration.items()
                    }
                },
            )
            # Donor files keep their source sequence in metadata for exclusion.
            for path in output.glob("*/donor_*.npz"):
                path.rename(output / path.name)
            write_json(
                output / "calibration_provenance.json",
                {
                    "dataset": args.dataset,
                    "selection_split": "dev",
                    "setup_sha256": digest(output / "setup.json"),
                    "sequences": args.sequence,
                },
            )
        write_json(
            output / "status.json",
            {
                "state": "complete",
                "stage": args.stage,
                "completed_units": completed,
                "elapsed_seconds": time.time() - started,
                "full_study_complete": False,
            },
        )
        run.summary["full_study_complete"] = False
        log_results_artifact(
            run,
            output,
            [
                str(p.relative_to(output))
                for p in output.rglob("*")
                if p.is_file() and "wandb" not in p.relative_to(output).parts
            ],
        )
    except BaseException as exc:
        write_json(
            output / "status.json",
            {
                "state": "failed",
                "error": str(exc),
                "elapsed_seconds": time.time() - started,
            },
        )
        log_results_artifact(
            run,
            output,
            [
                str(p.relative_to(output))
                for p in output.rglob("*.json*")
                if "wandb" not in p.relative_to(output).parts
            ],
        )
        run.finish(exit_code=1)
        raise
    run.finish()


if __name__ == "__main__":
    main()
