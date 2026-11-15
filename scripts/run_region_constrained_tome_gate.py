"""Full-video engineering gate for region-constrained ToMe memory reads.

Runs native SAM2 replay and two matched local-token-merging controls on seq2/3.
This script is an engineering gate, not a cohort efficacy runner.
"""

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.region_constrained_tome import NativeRegionConstrainedToMe, author_tome
from modeling.stat_calibrated_tome import StatisticallyCalibratedToMe
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import (
    build_author_predictor,
    tensor_hash,
)
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_soft_role_affinity_cv import score_frame
from scripts.run_regional_refinement import write


PROTOCOL = "protocols/region_constrained_tome_polypgen_v1.json"
NATIVE_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SEQUENCES = ("seq2", "seq3")
CONDITIONS = ("native", "region_constrained_tome", "generic_local_tome")
CALIBRATED_CONDITIONS = (
    "native",
    "region_constrained_tome",
    "generic_local_tome",
    "calibrated_region_constrained_tome",
    "calibrated_generic_local_tome",
)


def native_reference(native_run, results, protocol, name, frames):
    evidence = results["frame_evidence"].get(name)
    if not evidence:
        raise RuntimeError(f"Native reference has no frame evidence for {name}")
    frame_path = Path(evidence["path"])
    if not frame_path.is_file() or digest(frame_path) != evidence["sha256"]:
        raise RuntimeError(f"Native frame evidence digest mismatch for {name}")
    rows = json.loads(frame_path.read_text())
    receipt_path = frame_path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("sequence") != name
        or receipt.get("checkpoint_sha256") != protocol["model"]["checkpoint_sha256"]
        or receipt.get("manifest_sha256") != digest(ROOT / "data/polypgen_sequence.jsonl")
        or receipt.get("model_config") != protocol["model"]["configuration"]
        or receipt.get("num_frames") != len(frames)
    ):
        raise RuntimeError(f"Native receipt provenance differs for {name}")
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    if receipt.get("prompt") != prompt:
        raise RuntimeError(f"Native prompt index differs for {name}")
    expected_frames = list(range(prompt + 1, len(frames)))
    if [row["frame"] for row in rows] != expected_frames:
        raise RuntimeError(f"Native reference has incomplete or misordered targets: {name}")
    if receipt.get("native_frames_checked") != len(rows):
        raise RuntimeError(f"Native receipt target count differs for {name}")
    return prompt, {int(row["frame"]): row for row in rows}, {
        "frames_path": str(frame_path),
        "frames_sha256": evidence["sha256"],
        "receipt_path": str(receipt_path),
        "prompt": prompt,
        "num_frames": len(frames),
        "targets": len(rows),
    }


def build_model(checkpoint, config, repo):
    args = SimpleNamespace(
        method="sam2",
        device="cpu",
        checkpoint=checkpoint,
        model_config=config,
    )
    model = build_author_predictor(args, repo)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("SAM2 is not frozen in evaluation mode")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Engineering gate requires CPU FP32 SAM2")
    if len(model.memory_attention.layers) != 4:
        raise RuntimeError("Expected exactly four memory-attention layers")
    if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1:
        raise RuntimeError("SAM2 memory schedule differs from the native reference")
    if model.fill_hole_area != 0:
        raise RuntimeError("SAM2 postprocessing differs from the native reference")
    return model


@torch.inference_mode()
def run_condition(name, condition, frames, prompt, reference, checkpoint, config, repo, folder,
                  merge_plan_by_target=None):
    model = build_model(checkpoint, config, repo)
    parameter_before = parameter_digest(model)
    rows, calls, seconds = [], [], 0.0
    manager = None
    if condition == "generic_local_tome":
        manager = NativeRegionConstrainedToMe(
            model, mode="global", merge_plan_by_target=merge_plan_by_target
        )
    elif condition == "region_constrained_tome":
        manager = NativeRegionConstrainedToMe(model, mode="regional")
    elif condition == "calibrated_generic_local_tome":
        manager = StatisticallyCalibratedToMe(
            model, mode="global", merge_plan_by_target=merge_plan_by_target
        )
    elif condition == "calibrated_region_constrained_tome":
        manager = StatisticallyCalibratedToMe(model, mode="regional")

    with tempfile.TemporaryDirectory(prefix=f"frames_{name}_", dir=folder) as temp:
        state, observed_prompt = prepare(model, frames, Path(temp))
        if observed_prompt != prompt:
            raise RuntimeError(f"Prompt changed between reference and run for {name}")
        iterator = None
        started = time.monotonic()
        try:
            context = manager if manager is not None else _null_context()
            with context as observer:
                iterator = iter(model.propagate_in_video(state, start_frame_idx=prompt))
                expected = prompt
                while expected < len(frames):
                    frame, object_ids, logits = next(iterator)
                    if frame != expected or object_ids != [1]:
                        raise RuntimeError(f"Unexpected frame/object order: {name}/{frame}")
                    if logits.shape != (1, 1, *load_shape(frames[frame].mask_path)):
                        raise RuntimeError(f"SAM2 output shape changed at {name}/{frame}: {tuple(logits.shape)}")
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"Nonfinite propagated logits at {name}/{frame}")
                    expected += 1
                    if frame == prompt:
                        continue

                    stored = state["output_dict_per_obj"][0]["non_cond_frame_outputs"].get(frame)
                    if stored is None:
                        raise RuntimeError(f"SAM2 did not store the predicted frame {name}/{frame}")
                    for field in ("pred_masks", "maskmem_features", "object_score_logits"):
                        value = stored.get(field)
                        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                            raise RuntimeError(f"Nonfinite stored {field} at {name}/{frame}")
                    scored = score_frame(logits, frames, frame)
                    stored_sha = fingerprint(stored)["sha256"]
                    row = {
                        **scored,
                        "stored_sha256": stored_sha,
                        "object_score": float(stored["object_score_logits"].detach().cpu().item()),
                    }
                    if condition == "native":
                        expected_row = reference.get(frame)
                        if expected_row is None or any(
                            row[key] != expected_row[key]
                            for key in ("prediction_sha256", "stored_sha256", "present", "scores", "absent_fp")
                        ):
                            raise RuntimeError(f"Native replay differs from archived reference: {name}/{frame}")

                    if manager is not None:
                        frame_calls = [call for call in observer.calls if call["frame"] == frame]
                        if [call["layer"] for call in frame_calls] != [0, 1, 2, 3]:
                            raise RuntimeError(f"Expected one read from every layer at {name}/{frame}")
                        for call in frame_calls:
                            if call["spatial_key_mass_sum"] != float(call["spatial_keys_before"]):
                                raise RuntimeError(f"Spatial attention mass not conserved at {name}/{frame}")
                            if call["key_mass_sum"] != float(call["spatial_keys_before"] + call["pointer_keys"]):
                                raise RuntimeError(f"Total attention mass not conserved at {name}/{frame}")
                            if call["spatial_keys_after"] >= call["spatial_keys_before"]:
                                raise RuntimeError(f"ToMe did not reduce spatial keys at {name}/{frame}")
                            if call["pointer_keys"] < 0:
                                raise RuntimeError("Negative pointer count")
                        row["attention_reads"] = frame_calls
                        calls.extend(frame_calls)

                    rows.append(row)
                    write(folder / "progress.json", {
                        "sequence": name,
                        "condition": condition,
                        "frame": frame,
                        "targets_done": len(rows),
                        "targets_total": len(reference),
                    })
                    print(f"[{condition}] {name}: {frame}/{len(frames)-1}", flush=True)
                seconds = time.monotonic() - started
        finally:
            if iterator is not None:
                iterator.close()
            model.reset_state(state)

    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
        raise RuntimeError(f"Incomplete trajectory for {condition}/{name}")
    parameter_after = parameter_digest(model)
    if parameter_after != parameter_before:
        raise RuntimeError(f"SAM2 parameters changed during {condition}/{name}")
    if manager is not None and len(calls) != len(rows) * 4:
        raise RuntimeError(f"Unexpected total attention-layer call count for {condition}/{name}")
    del model
    gc.collect()
    return rows, {
        "sequence": name,
        "condition": condition,
        "targets": len(rows),
        "attention_layer_calls": len(calls),
        "wall_seconds": seconds,
        "parameter_digest": parameter_before,
        "parameters_unchanged": True,
    }


def load_shape(image_path):
    from PIL import Image

    with Image.open(image_path) as image:
        return image.height, image.width


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def compare_matched_budgets(results, conditions=CONDITIONS):
    comparisons = []
    pairs = [("generic_local_tome", "region_constrained_tome")]
    calibration_pairs = []
    if "calibrated_generic_local_tome" in conditions:
        pairs.append(("calibrated_generic_local_tome", "calibrated_region_constrained_tome"))
        calibration_pairs = [
            ("calibrated_region_constrained_tome", "region_constrained_tome"),
            ("calibrated_generic_local_tome", "generic_local_tome"),
        ]
        pairs.extend(calibration_pairs)
    for name in SEQUENCES:
        for generic_name, regional_name in pairs:
            generic = {row["frame"]: row for row in results[name][generic_name]}
            regional = {row["frame"]: row for row in results[name][regional_name]}
            if generic.keys() != regional.keys():
                raise RuntimeError(f"Matched controls differ in target frames for {name}")
            for frame in generic:
                gcalls = generic[frame]["attention_reads"]
                rcalls = regional[frame]["attention_reads"]
                if len(gcalls) != len(rcalls) or len(gcalls) != 4:
                    raise RuntimeError(f"Matched controls have different layer calls at {name}/{frame}")
                for gcall, rcall in zip(gcalls, rcalls, strict=True):
                    if gcall["layer"] != rcall["layer"]:
                        raise RuntimeError("Matched controls have different layer order")
                    if (generic_name, regional_name) in calibration_pairs:
                        assignment = gcall.get("merge_assignment_sha256")
                        reference_assignment = rcall.get("merge_assignment_sha256")
                        if (
                            not isinstance(assignment, str)
                            or len(assignment) != 64
                            or any(char not in "0123456789abcdef" for char in assignment)
                            or assignment != reference_assignment
                        ):
                            raise RuntimeError(
                                f"Missing or unequal exact calibration merge assignments at {name}/{frame}"
                            )
                    if gcall["pointer_keys"] != rcall["pointer_keys"]:
                        raise RuntimeError("Matched controls have different pointer counts")
                    g_frames = {
                        row["frame"]: row["retained_spatial_keys"]
                        for row in gcall["frames"]
                    }
                    r_frames = {
                        row["frame"]: row["retained_spatial_keys"]
                        for row in rcall["frames"]
                    }
                    if g_frames != r_frames:
                        raise RuntimeError(f"Spatial key budgets differ by memory frame at {name}/{frame}")
                    g_windows = {
                        (memory_frame["frame"], row["window"]): row["retained_spatial_keys"]
                        for memory_frame in gcall["frames"]
                        for row in memory_frame["windows"]
                    }
                    r_windows = {
                        (memory_frame["frame"], row["window"]): row["retained_spatial_keys"]
                        for memory_frame in rcall["frames"]
                        for row in memory_frame["windows"]
                    }
                    if g_windows != r_windows:
                        raise RuntimeError(f"Spatial key budgets differ by local window at {name}/{frame}")
                    comparisons.append({
                        "sequence": name,
                        "target_frame": frame,
                        "generic_condition": generic_name,
                        "regional_condition": regional_name,
                        "layer": gcall["layer"],
                        "spatial_keys_after": gcall["spatial_keys_after"],
                        "pointer_keys": gcall["pointer_keys"],
                        "frame_key_counts_match": True,
                    })
    return comparisons


def matched_plan_from_regional(rows):
    """Freeze regional per-window merge counts for the generic matched control."""
    plan = {}
    for row in rows:
        reads = row["attention_reads"]
        if [item["layer"] for item in reads] != [0, 1, 2, 3]:
            raise RuntimeError("Cannot derive a matched plan without all four regional layers")
        first = reads[0]["frames"]
        expected = {
            item["frame"]: {
                window["window"]: window["merges"] for window in item["windows"]
            }
            for item in first
        }
        for read in reads[1:]:
            observed = {
                item["frame"]: {
                    window["window"]: window["merges"] for window in item["windows"]
                }
                for item in read["frames"]
            }
            if observed != expected:
                raise RuntimeError("Regional per-layer merge budgets differ")
        plan[row["frame"]] = expected
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stat-calibrated", action="store_true",
        help="Also gate calibrated generic/regional ToMe against their matched controls.",
    )
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        conditions = CALIBRATED_CONDITIONS if args.stat_calibrated else CONDITIONS
        if args.stat_calibrated:
            if not StatisticallyCalibratedToMe.supports_frozen_assignment_replay:
                raise RuntimeError(
                    "Calibration gate blocked: exact cross-arm merge-assignment replay is not implemented"
                )
            # Use the clean pinned mirror. The user's working checkout has
            # unrelated edits to image_encoder.py and must not enter this gate.
            import scripts.run_author_baseline as baseline
            baseline.REPOSITORIES["sam2"] = (
                "external/sam2_diagnostic_reference", NATIVE_REVISION
            )
        protocol_path = (
            ROOT / "protocols/stat_calibrated_tome_gate_v1.json"
            if args.stat_calibrated else ROOT / PROTOCOL
        )
        protocol = json.loads(protocol_path.read_text()) if args.stat_calibrated else json.loads(
            (ROOT / PROTOCOL).read_text()
        )
        if args.checkpoint.is_file() is False:
            raise FileNotFoundError(args.checkpoint)
        checkpoint_sha = digest(args.checkpoint)
        if checkpoint_sha != protocol["model"]["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint differs from the frozen gate protocol")
        manifest = ROOT / protocol["scope"]["manifest"]
        manifest_sha = digest(manifest)
        source_sha = digest(ROOT / "external/ToMe/tome/merge.py")
        if source_sha != protocol["prior_work"]["merge_source_sha256"]:
            raise RuntimeError("ToMe merge source differs from the frozen protocol")
        author_tome()
        repo, revision = activate_author_repo()
        if revision != protocol["model"]["revision"] or revision != NATIVE_REVISION:
            raise RuntimeError("SAM2 author repository differs from the frozen protocol")
        if not args.native_run.is_dir():
            raise FileNotFoundError(args.native_run)
        results_path = args.native_run / "native_results.json"
        native_results = json.loads(results_path.read_text())
        if native_results.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("Archived native reference used a different checkpoint")
        videos = load_manifest(manifest, "dev")
        write(output / "protocol.json", protocol)
        write(output / "setup.json", {
            "protocol_sha256": digest(protocol_path),
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "native_results_sha256": digest(results_path),
            "sam2_revision": revision,
            "tome_revision": protocol["prior_work"]["repository_revision"],
            "tome_merge_source_sha256": source_sha,
            "gate_script_sha256": digest(Path(__file__)),
            "method_module_sha256": digest(ROOT / "modeling/region_constrained_tome.py"),
            "calibration_module_sha256": digest(ROOT / "modeling/stat_calibrated_tome.py"),
            "device": "cpu",
            "dtype": "float32",
        })
        results = {name: {} for name in SEQUENCES}
        receipts = []
        settings = []
        for name in SEQUENCES:
            frames = videos[name]
            prompt, reference, ref_receipt = native_reference(
                args.native_run, native_results, protocol, name, frames
            )
            receipts.append(ref_receipt)
            merge_plan_by_target = None
            for condition in conditions:
                if condition == "calibrated_region_constrained_tome":
                    # This arm establishes its own role-constrained per-window
                    # quotas; its following generic control reuses them.
                    merge_plan_by_target = None
                folder = output / condition / name
                folder.mkdir(parents=True, exist_ok=False)
                rows, summary = run_condition(
                    name,
                    condition,
                    frames,
                    prompt,
                    reference,
                    args.checkpoint,
                    protocol["model"]["configuration"],
                    repo,
                    folder,
                    merge_plan_by_target=merge_plan_by_target,
                )
                results[name][condition] = rows
                write(folder / "frames.json", rows)
                write(folder / "summary.json", summary)
                settings.append(summary)
                if condition in (
                    "region_constrained_tome",
                    "calibrated_region_constrained_tome",
                ):
                    merge_plan_by_target = matched_plan_from_regional(rows)
                    (output / "matched_merge_plan").mkdir(parents=True, exist_ok=True)
                    write(output / "matched_merge_plan" / f"{name}.json", {
                        str(target): {
                            str(memory_frame): {str(window): count for window, count in windows.items()}
                            for memory_frame, windows in memory_plan.items()
                        }
                        for target, memory_plan in merge_plan_by_target.items()
                    })
        matched = compare_matched_budgets(results, conditions)
        summary = {
            "passed": True,
            "scope": "seq2/seq3 engineering gate only; no efficacy decision",
            "conditions": list(conditions),
            "sequences": {
                name: {condition: len(results[name][condition]) for condition in conditions}
                for name in SEQUENCES
            },
            "all_native_frames_match_archived_logits_and_stored_outputs": True,
            "all_modified_reads_cover_four_layers": True,
            "matched_budget_per_frame_and_layer": True,
            "calibrated_merge_assignments_identical_to_uncalibrated": (
                True if args.stat_calibrated else None
            ),
            "pointer_key_counts_unchanged_between_controls": True,
            "all_attention_mass_conserved": True,
            "all_model_parameters_unchanged": True,
            "native_reference_receipts": receipts,
            "runs": settings,
            "matched_budget_checks": len(matched),
            "tome_repository_revision": protocol["prior_work"]["repository_revision"],
            "tome_source_sha256": source_sha,
            "sam2_revision": revision,
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
        }
        write(output / "gate.json", summary)
        write(output / "matched_budget_checks.json", matched)
        write(output / "completion.json", {"passed": True, "sequences": list(SEQUENCES), "conditions": list(conditions)})
    except BaseException as error:
        write(output / "failure.json", {"passed": False, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
