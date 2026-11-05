"""Seq2/seq3 engineering gate for continuous mask-aware ToMe memory reads.

This adapts the existing ToMe gate workflow. It verifies replay and matched
budgets only; it is not a Dice efficacy experiment.
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
from modeling.region_constrained_tome import author_tome
from modeling.soft_mask_aware_tome import NativeSoftMaskAwareToMe
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_region_constrained_tome_gate import load_shape, native_reference, _null_context
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import score_frame


PROTOCOL = "protocols/soft_mask_aware_tome_polypgen_v1.json"
SEQUENCES = ("seq2", "seq3")
CONDITIONS = ("native", "generic_local_tome", "soft_mask_aware_tome")


def build_model(checkpoint, config, repo):
    args = SimpleNamespace(
        method="sam2", device="cpu", checkpoint=checkpoint, model_config=config
    )
    model = build_author_predictor(args, repo)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("SAM2 must be frozen and in evaluation mode")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("The gate requires CPU FP32 SAM2")
    if len(model.memory_attention.layers) != 4:
        raise RuntimeError("Expected all four native memory-attention layers")
    if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1:
        raise RuntimeError("Native SAM2 memory schedule differs from the frozen protocol")
    if model.fill_hole_area != 0:
        raise RuntimeError("Native SAM2 postprocessing differs from the reference")
    return model


@torch.inference_mode()
def run_condition(name, condition, frames, prompt, reference, checkpoint, config, repo, folder):
    model = build_model(checkpoint, config, repo)
    before = parameter_digest(model)
    observer = None
    if condition == "generic_local_tome":
        observer = NativeSoftMaskAwareToMe(model, mode="generic")
    elif condition == "soft_mask_aware_tome":
        observer = NativeSoftMaskAwareToMe(model, mode="soft_mask_aware")

    rows, seconds = [], 0.0
    with tempfile.TemporaryDirectory(prefix=f"soft_tome_{name}_", dir=folder) as temp:
        state, observed_prompt = prepare(model, frames, Path(temp))
        if observed_prompt != prompt:
            raise RuntimeError(f"Prompt changed between reference and gate for {name}")
        iterator = None
        started = time.monotonic()
        try:
            context = observer if observer is not None else _null_context()
            with context:
                iterator = iter(model.propagate_in_video(state, start_frame_idx=prompt))
                expected = prompt
                while expected < len(frames):
                    frame, object_ids, logits = next(iterator)
                    if frame != expected or object_ids != [1]:
                        raise RuntimeError(f"Unexpected frame/object order: {name}/{frame}")
                    if logits.shape != (1, 1, *load_shape(frames[frame].mask_path)):
                        raise RuntimeError(f"Prediction shape changed at {name}/{frame}")
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"Nonfinite prediction at {name}/{frame}")
                    expected += 1
                    if frame == prompt:
                        continue

                    stored = state["output_dict_per_obj"][0]["non_cond_frame_outputs"].get(frame)
                    if stored is None:
                        raise RuntimeError(f"Missing stored prediction at {name}/{frame}")
                    for field in ("pred_masks", "maskmem_features", "object_score_logits"):
                        value = stored.get(field)
                        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                            raise RuntimeError(f"Nonfinite stored {field} at {name}/{frame}")
                    row = {
                        **score_frame(logits, frames, frame),
                        "stored_sha256": fingerprint(stored)["sha256"],
                        "object_score": float(stored["object_score_logits"].detach().cpu().item()),
                    }
                    if condition == "native":
                        expected_row = reference.get(frame)
                        if expected_row is None or any(
                            row[key] != expected_row[key]
                            for key in (
                                "prediction_sha256", "stored_sha256", "present", "scores", "absent_fp"
                            )
                        ):
                            raise RuntimeError(f"Native replay differs from archived output: {name}/{frame}")

                    if observer is not None:
                        reads = [call for call in observer.calls if call["frame"] == frame]
                        if [call["layer"] for call in reads] != [0, 1, 2, 3]:
                            raise RuntimeError(f"Expected exactly four attention reads at {name}/{frame}")
                        for call in reads:
                            if call["spatial_key_mass_sum"] != float(call["spatial_keys_before"]):
                                raise RuntimeError("Spatial attention mass was not conserved")
                            if call["key_mass_sum"] != float(
                                call["spatial_keys_before"] + call["pointer_keys"]
                            ):
                                raise RuntimeError("Total attention mass was not conserved")
                            if call["spatial_keys_after"] >= call["spatial_keys_before"]:
                                raise RuntimeError("ToMe did not compress spatial keys")
                            if [source["frame"] for source in call["mask_sources"]] != [
                                source["frame"] for source in call["frames"]
                            ]:
                                raise RuntimeError("Stored-mask source frames do not match K/V memory spans")
                            for source, memory_frame in zip(
                                call["mask_sources"], call["frames"], strict=True
                            ):
                                expected_mask_shape = [model.image_size, model.image_size]
                                if source["stored_mask_field"] != "pred_masks_high_res":
                                    raise RuntimeError("Occupancy did not use SAM2's stored high-resolution mask")
                                if source["stored_mask_shape"] != expected_mask_shape:
                                    raise RuntimeError("Stored high-resolution mask differs from model image size")
                                token_count = sum(
                                    window["original_spatial_keys"]
                                    for window in memory_frame["windows"]
                                )
                                if source["occupancy_tokens"] != token_count:
                                    raise RuntimeError("Stored-mask occupancy does not span the memory K/V keys")
                                grid_shape = source["memory_grid_shape"]
                                if (
                                    len(grid_shape) != 2
                                    or min(grid_shape) <= 0
                                    or grid_shape[0] * grid_shape[1] != token_count
                                ):
                                    raise RuntimeError("Occupancy pooling grid does not match the memory K/V grid")
                                if source["occupancy_pool_kernel"] != [
                                    expected_mask_shape[0] // grid_shape[0],
                                    expected_mask_shape[1] // grid_shape[1],
                                ]:
                                    raise RuntimeError("Occupancy pooling footprint does not match mask-to-memory geometry")
                        row["attention_reads"] = reads

                    rows.append(row)
                    write(folder / "progress.json", {
                        "sequence": name,
                        "condition": condition,
                        "frame": frame,
                        "targets_done": len(rows),
                        "targets_total": len(reference),
                    })
                    print(f"[{condition}] {name}: {frame}/{len(frames) - 1}", flush=True)
                seconds = time.monotonic() - started
        finally:
            if iterator is not None:
                iterator.close()
            model.reset_state(state)

    if [row["frame"] for row in rows] != list(range(prompt + 1, len(frames))):
        raise RuntimeError(f"Incomplete trajectory for {condition}/{name}")
    after = parameter_digest(model)
    if after != before:
        raise RuntimeError(f"SAM2 parameters changed during {condition}/{name}")
    calls = sum(len(row.get("attention_reads", [])) for row in rows)
    if observer is not None and calls != len(rows) * 4:
        raise RuntimeError(f"Unexpected attention-layer call count for {condition}/{name}")
    summary = {
        "sequence": name,
        "condition": condition,
        "targets": len(rows),
        "attention_layer_calls": calls,
        "wall_seconds": seconds,
        "parameter_digest": before,
        "parameters_unchanged": True,
    }
    del model
    gc.collect()
    return rows, summary


def compare_budgets(results):
    checks = []
    for sequence in SEQUENCES:
        generic = {row["frame"]: row for row in results[sequence]["generic_local_tome"]}
        aware = {row["frame"]: row for row in results[sequence]["soft_mask_aware_tome"]}
        if generic.keys() != aware.keys():
            raise RuntimeError(f"Generic and mask-aware target frames differ: {sequence}")
        for frame in generic:
            generic_reads = generic[frame]["attention_reads"]
            aware_reads = aware[frame]["attention_reads"]
            for generic_call, aware_call in zip(generic_reads, aware_reads, strict=True):
                if generic_call["layer"] != aware_call["layer"]:
                    raise RuntimeError("Matched arms have different layer order")
                if generic_call["pointer_keys"] != aware_call["pointer_keys"]:
                    raise RuntimeError("Matched arms changed the pointer-key count")
                generic_windows = {
                    (memory["frame"], window["window"]): window["retained_spatial_keys"]
                    for memory in generic_call["frames"] for window in memory["windows"]
                }
                aware_windows = {
                    (memory["frame"], window["window"]): window["retained_spatial_keys"]
                    for memory in aware_call["frames"] for window in memory["windows"]
                }
                if generic_windows != aware_windows:
                    raise RuntimeError(f"Generic and mask-aware budgets differ at {sequence}/{frame}")
                checks.append({
                    "sequence": sequence,
                    "target_frame": frame,
                    "layer": generic_call["layer"],
                    "window_budgets_match": True,
                    "pointer_count_matches": True,
                })
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        protocol = json.loads((ROOT / PROTOCOL).read_text())
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        checkpoint_sha = digest(args.checkpoint)
        if checkpoint_sha != protocol["model"]["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint differs from frozen soft-ToMe protocol")
        manifest = ROOT / protocol["scope"]["manifest"]
        manifest_sha = digest(manifest)
        tome_source_sha = digest(ROOT / "external/ToMe/tome/merge.py")
        if tome_source_sha != protocol["prior_work"]["source_sha256"]:
            raise RuntimeError("Pinned ToMe source differs from the protocol")
        author_tome()
        repo, revision = activate_author_repo()
        if revision != protocol["model"]["revision"]:
            raise RuntimeError("SAM2 author repository differs from the protocol")
        if not args.native_run.is_dir():
            raise FileNotFoundError(args.native_run)
        native_path = args.native_run / "native_results.json"
        native_results = json.loads(native_path.read_text())
        if native_results.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("Archived native run used a different checkpoint")
        videos = load_manifest(manifest, "dev")
        write(output / "protocol.json", protocol)
        write(output / "setup.json", {
            "protocol_sha256": digest(ROOT / PROTOCOL),
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
            "native_results_sha256": digest(native_path),
            "sam2_revision": revision,
            "tome_revision": protocol["prior_work"]["repository_revision"],
            "tome_merge_source_sha256": tome_source_sha,
            "gate_script_sha256": digest(Path(__file__)),
            "method_module_sha256": digest(ROOT / "modeling/soft_mask_aware_tome.py"),
            "source_file_digests": {
                source: digest(ROOT / source)
                for source in protocol["engineering_gate"]["source_files_to_hash"]
            },
            "device": "cpu",
            "dtype": "float32",
        })
        results, summaries, references = {name: {} for name in SEQUENCES}, [], []
        for name in SEQUENCES:
            frames = videos[name]
            prompt, reference, receipt = native_reference(
                args.native_run, native_results, protocol, name, frames
            )
            references.append(receipt)
            for condition in CONDITIONS:
                folder = output / condition / name
                folder.mkdir(parents=True, exist_ok=False)
                rows, summary = run_condition(
                    name, condition, frames, prompt, reference, args.checkpoint,
                    protocol["model"]["configuration"], repo, folder,
                )
                results[name][condition] = rows
                write(folder / "frames.json", rows)
                write(folder / "summary.json", summary)
                summaries.append(summary)
        matched = compare_budgets(results)
        gate = {
            "passed": True,
            "scope": "seq2/seq3 engineering gate only; not an efficacy conclusion",
            "conditions": list(CONDITIONS),
            "sequences": {
                name: {condition: len(results[name][condition]) for condition in CONDITIONS}
                for name in SEQUENCES
            },
            "native_replay_matches_archived_predictions_and_stored_outputs": True,
            "all_four_memory_attention_layers_intercepted": True,
            "stored_mask_frames_and_occupancy_spans_match_native_memory_selection": True,
            "mask_aware_and_generic_budgets_match_per_window": True,
            "pointers_unchanged_and_mass_conserved": True,
            "all_sam2_parameters_unchanged": True,
            "native_reference_receipts": references,
            "runs": summaries,
            "matched_budget_checks": len(matched),
            "tome_repository_revision": protocol["prior_work"]["repository_revision"],
            "tome_source_sha256": tome_source_sha,
            "sam2_revision": revision,
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha,
        }
        write(output / "gate.json", gate)
        write(output / "matched_budget_checks.json", matched)
        write(output / "completion.json", {
            "passed": True, "sequences": list(SEQUENCES), "conditions": list(CONDITIONS)
        })
    except BaseException as error:
        write(output / "failure.json", {"passed": False, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
