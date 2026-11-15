"""Real-clip engineering gate for representative-position long-history ToMe.

This gate compares position-mean and representative-position aggregation on
PolypGen seq3. It checks plumbing and deterministic replay only; it is not an
efficacy runner.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.position_consistent_history_tome import (
    PositionConsistentLongHistoryToMe,
    build_fixed_budget_merge_plan,
)
from modeling.wider_history_tome import MAP_TOKENS, MAX_NONCOND_MAPS
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, prompt_index, validate_output
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_position_consistent_tome_gate import (
    build_model,
    load_shape,
    native_reference,
)
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import score_frame
from scripts.run_wider_history_tome_gate import (
    CHECKPOINT_DEFAULT,
    SAM2_REFERENCE,
    activate_reference_repo,
)


GATE_PROTOCOL = ROOT / "protocols/position_consistent_long_history_tome_gate_v2.json"
NATIVE_PROTOCOL = ROOT / "protocols/position_consistent_region_tome_polypgen_v2.json"
NATIVE_SETUP = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-"
    "phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/"
    "143_official_native_feature_audit_v1/full/setup.json"
)
NATIVE_RUN = NATIVE_SETUP.parent
CHECKPOINT_DEFAULT = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-"
    "phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/"
    "assets/sam2.1_hiera_tiny.pt"
)
SEQUENCE = "seq3"
CONDITIONS = {"position_mean": "mean", "position_representative": "representative"}
SOURCE_PATHS = (
    "protocols/position_consistent_long_history_tome_gate_v1.json",
    "protocols/position_consistent_long_history_tome_gate_v2.json",
    "protocols/position_consistent_kv_matching_gate_v1.json",
    "protocols/region_constrained_tome_polypgen_v1.json",
    "modeling/position_consistent_history_tome.py",
    "modeling/position_consistent_tome.py",
    "modeling/wider_history_tome.py",
    "scripts/run_position_consistent_history_tome_gate.py",
    "scripts/run_position_consistent_tome_gate.py",
    "scripts/run_wider_history_tome_gate.py",
    "scripts/run_author_baseline.py",
    "scripts/run_cfbi_region_pilot.py",
    "scripts/run_soft_role_affinity_cv.py",
    "scripts/memory_token_io.py",
    "datasets/video.py",
    "external/ToMe/tome/merge.py",
)


def source_receipt():
    return {path: digest(ROOT / path) for path in SOURCE_PATHS}


def _validate_attention_read(call, position_mode, matching_mode):
    if call.get("position_mode") != position_mode:
        raise RuntimeError("Attention read used the wrong position aggregation mode")
    if call.get("matching_mode") != matching_mode:
        raise RuntimeError("Attention read used the wrong ToMe matching metric")
    if call["spatial_key_mass_sum"] != float(call["spatial_keys_before"]):
        raise RuntimeError("Spatial mass is not conserved")
    if call["key_mass_sum"] != float(call["spatial_keys_before"] + call["pointer_keys"]):
        raise RuntimeError("Pointer-inclusive mass is not conserved")

    frames = call["frames"]
    spans, cursor = [], 0
    for row in frames:
        if row["input_spatial_keys"] != MAP_TOKENS:
            raise RuntimeError("Expected a dense 64x64 source memory map")
        spans.append({
            "frame": int(row["frame"]),
            "t_pos": int(row["t_pos"]),
            "start": cursor,
            "stop": cursor + int(row["input_spatial_keys"]),
        })
        cursor += int(row["input_spatial_keys"])
        if len(row["windows"]) != 256:
            raise RuntimeError("Expected exactly 256 4x4 windows per memory frame")
        if any(window["input_keys"] != 16 or not window["partition_valid"] for window in row["windows"]):
            raise RuntimeError("Invalid per-window source partition")

    if cursor != call["spatial_keys_before"]:
        raise RuntimeError("Frame spans do not cover the spatial keys")
    plan = build_fixed_budget_merge_plan(
        spans, max_noncond_maps=10, window_size=4, allocation="uniform"
    )
    for span, row in zip(spans, frames, strict=True):
        frame_plan = plan[span["frame"]]
        observed = {int(window["window"]): int(window["merges"]) for window in row["windows"]}
        if observed != frame_plan:
            raise RuntimeError("Observed local ToMe counts differ from the frozen allocation")
        retained = span["stop"] - span["start"] - sum(frame_plan.values())
        if int(row["output_spatial_keys"]) != retained:
            raise RuntimeError("Per-frame retained-key count is incorrect")

    noncond = sum(span["t_pos"] > 0 for span in spans)
    expected_total = 4096 + min(noncond, 6) * 4096 if noncond <= 6 else 4096 + 24576
    if call["spatial_keys_after"] != expected_total:
        raise RuntimeError("Read does not follow the fixed native spatial-token budget")
    hashes = {
        key: call[key]
        for key in (
            "input_memory_sha256",
            "input_position_sha256",
            "output_memory_sha256",
            "output_position_sha256",
            "attention_bias_sha256",
            "rope_positions_sha256",
        )
        if key in call
    }
    return {
        "layer": int(call["layer"]),
        "position_mode": call["position_mode"],
        "matching_mode": call["matching_mode"],
        "spatial_keys_before": int(call["spatial_keys_before"]),
        "spatial_keys_after": int(call["spatial_keys_after"]),
        "pointer_keys": int(call["pointer_keys"]),
        "spatial_key_mass_sum": float(call["spatial_key_mass_sum"]),
        "key_mass_sum": float(call["key_mass_sum"]),
        "tensor_hashes": hashes,
        "noncond_maps": noncond,
        "frames": [
            {
                "frame": int(row["frame"]),
                "t_pos": int(row["t_pos"]),
                "input_spatial_keys": int(row["input_spatial_keys"]),
                "output_spatial_keys": int(row["output_spatial_keys"]),
                "quota_sha256": hashlib.sha256(json.dumps(
                    [int(window["merges"]) for window in row["windows"]],
                    separators=(",", ":"),
                ).encode()).hexdigest(),
            }
            for row in frames
        ],
    }


def trial(args):
    # Match the archived native run's CPU runtime. The first gate attempt
    # used one thread plus deterministic-algorithm dispatch; SAM2's BF16
    # memory encoder then differed while predicted logits stayed identical.
    torch.set_num_threads(2)
    torch.manual_seed(20261021)
    torch.use_deterministic_algorithms(False)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    gate = json.loads(args.gate_protocol.read_text())
    if gate.get("status") != "engineering only; no efficacy inference authorized":
        raise RuntimeError("Gate protocol status is not the frozen engineering-only scope")
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    checkpoint_sha = digest(args.checkpoint)
    if checkpoint_sha != native_protocol["source_and_assets"]["sam2_checkpoint_sha256"]:
        raise RuntimeError("Checkpoint does not match the frozen native reference")
    manifest = ROOT / "data/polypgen_sequence.jsonl"
    videos = load_manifest(manifest, "dev")
    frames = videos[SEQUENCE]
    prompt = prompt_index(frames, {"initialization": "first_labeled_frame"})
    final_target = len(frames) - 1
    native_results = json.loads((NATIVE_RUN / "native_results.json").read_text())
    native_prompt, native_rows, native_receipt = native_reference(
        NATIVE_RUN, native_results, native_protocol, SEQUENCE, frames
    )
    if native_prompt != prompt:
        raise RuntimeError("Archived native prompt differs from manifest-derived prompt")

    inputs = {}
    for frame in frames[prompt:final_target + 1]:
        inputs[f"image:{frame.frame_index}"] = digest(frame.image_path)
        inputs[f"mask:{frame.frame_index}"] = digest(frame.mask_path)
    revision = activate_reference_repo()
    repo = SAM2_REFERENCE
    if revision != native_protocol["source_and_assets"]["sam2_revision"]:
        raise RuntimeError("SAM2 repository revision differs from archived native reference")
    before_sources = source_receipt()
    write(output / "setup.json", {
        "gate_protocol": str(args.gate_protocol),
        "gate_protocol_sha256": digest(args.gate_protocol),
        "native_protocol_sha256": digest(NATIVE_PROTOCOL),
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": digest(manifest),
        "native_results_sha256": digest(NATIVE_RUN / "native_results.json"),
        "native_reference_receipt": native_receipt,
        "sam2_revision": revision,
        "tome_revision": "af95e4b1befa172dadccd8c81e223b10090f9579",
        "sequence": SEQUENCE,
        "prompt": prompt,
        "target_frames": list(range(prompt + 1, final_target + 1)),
        "inputs": inputs,
        "sources": before_sources,
        "condition": args.condition,
        "position_mode": CONDITIONS[args.condition],
        "matching_mode": args.matching_mode,
        "device": "cpu",
        "dtype": "float32",
        "runtime": {
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "seed": 20261021,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    })

    model = build_model(
        args.checkpoint,
        native_protocol["source_and_assets"]["sam2_config"],
        repo,
    )
    parameters_before = parameter_digest(model)
    native_num_maskmem = model.num_maskmem
    native_tpos = model.maskmem_tpos_enc.detach().clone()
    candidate = PositionConsistentLongHistoryToMe(
        model,
        max_noncond_maps=10,
        allocation="uniform",
        position_mode=CONDITIONS[args.condition],
        matching_mode=args.matching_mode,
        provenance_target=min(final_target, prompt + 8),
        hash_tensors_target=min(final_target, prompt + 8),
    )

    rows, trace = [], []
    with tempfile.TemporaryDirectory(prefix=f"history_{SEQUENCE}_", dir=output) as temp:
        state, observed_prompt = prepare(model, frames, Path(temp))
        if observed_prompt != prompt:
            raise RuntimeError("Predictor prompt differs from frozen manifest prompt")
        iterator = None
        try:
            with candidate as observer:
                iterator = iter(model.propagate_in_video(
                    state,
                    start_frame_idx=prompt,
                    max_frame_num_to_track=final_target - prompt,
                ))
                expected = prompt
                while expected <= final_target:
                    frame, object_ids, logits = next(iterator)
                    if frame != expected or object_ids != [1]:
                        raise RuntimeError(f"Unexpected frame/object order at {frame}")
                    if logits.shape != (1, 1, *load_shape(frames[frame].mask_path)):
                        raise RuntimeError(f"Unexpected mask shape at target {frame}: {tuple(logits.shape)}")
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"Nonfinite output logits at target {frame}")
                    expected += 1
                    if frame == prompt:
                        continue
                    stored = state["output_dict_per_obj"][0]["non_cond_frame_outputs"].get(frame)
                    if stored is None:
                        raise RuntimeError(f"No predicted memory write for target {frame}")
                    for key in ("pred_masks", "maskmem_features", "object_score_logits"):
                        value = stored.get(key)
                        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                            raise RuntimeError(f"Nonfinite stored {key} at target {frame}")
                    row = {
                        **score_frame(logits, frames, frame),
                        "stored_sha256": fingerprint(stored)["sha256"],
                        "object_score": float(stored["object_score_logits"].detach().cpu().item()),
                    }
                    frame_calls = [call for call in observer.calls if call["frame"] == frame]
                    if [call["layer"] for call in frame_calls] != [0, 1, 2, 3]:
                        raise RuntimeError(f"Memory attention layers missing at target {frame}")
                    read_summaries = [
                        _validate_attention_read(
                            call, CONDITIONS[args.condition], args.matching_mode
                        )
                        for call in frame_calls
                    ]
                    if any(read["noncond_maps"] > 6 for read in read_summaries):
                        row["compressed_history"] = True
                        if frame == min(final_target, prompt + 8):
                            required_hashes = {
                                "input_memory_sha256", "input_position_sha256",
                                "output_memory_sha256", "output_position_sha256",
                                "attention_bias_sha256", "rope_positions_sha256",
                            }
                            if any(set(read["tensor_hashes"]) != required_hashes for read in read_summaries):
                                raise RuntimeError("Missing per-layer compressed tensor hashes at first compressed target")
                    else:
                        row["compressed_history"] = False
                        reference = native_rows.get(frame)
                        if reference is None:
                            raise RuntimeError(f"Native reference is missing target {frame}")
                        parity_fields = ("prediction_sha256", "stored_sha256", "present", "scores", "absent_fp")
                        mismatches = {
                            key: {"candidate": row.get(key), "native": reference.get(key)}
                            for key in parity_fields
                            if row.get(key) != reference.get(key)
                        }
                        if mismatches:
                            write(output / "precompression_mismatch.json", {
                                "frame": frame,
                                "mismatches": mismatches,
                                "stored_field_hashes": {
                                    key: fingerprint(stored.get(key))
                                    for key in ("pred_masks", "maskmem_features", "object_score_logits", "maskmem_pos_enc")
                                    if key in stored
                                },
                                "native_reference": reference,
                            })
                            raise RuntimeError(
                                f"Pre-compression trajectory differs from native at {frame}: "
                                f"{', '.join(mismatches)}"
                            )
                        row["exact_native_prefix"] = True
                    row["attention_reads"] = read_summaries
                    trace.extend({"target_frame": frame, **read} for read in read_summaries)
                    rows.append(row)
                    write(output / "progress.json", {
                        "status": "running",
                        "condition": args.condition,
                        "sequence": SEQUENCE,
                        "target": frame,
                        "targets_done": len(rows),
                        "targets_total": final_target - prompt,
                        "compressed_targets": sum(item["compressed_history"] for item in rows),
                    })
                    print(f"[{args.condition}] {SEQUENCE}: {frame}/{final_target}", flush=True)
        finally:
            if iterator is not None:
                iterator.close()
            model.reset_state(state)

    if [row["frame"] for row in rows] != list(range(prompt + 1, final_target + 1)):
        raise RuntimeError("The full seq3 post-prompt trajectory was not completed")
    if not any(row["compressed_history"] for row in rows):
        raise RuntimeError("The clip did not reach the compressed-history path")
    parameters_after = parameter_digest(model)
    if parameters_after != parameters_before:
        raise RuntimeError("SAM2 parameters changed during the real-clip gate")
    if model.num_maskmem != native_num_maskmem or not torch.equal(model.maskmem_tpos_enc.detach(), native_tpos):
        raise RuntimeError("Extended temporal state was not restored")
    if source_receipt() != before_sources:
        raise RuntimeError("Frozen source changed during the real-clip gate")

    provenance = candidate.provenance_reads
    if not provenance:
        raise RuntimeError("No cluster provenance was captured for a compressed target")
    write(output / "frames.json", rows)
    write(output / "attention_trace.json", trace)
    write(output / "first_compressed_target_provenance.json", provenance)
    write(output / "completion.json", {
        "passed": True,
        "condition": args.condition,
        "matching_mode": args.matching_mode,
        "sequence": SEQUENCE,
        "targets": len(rows),
        "compressed_targets": sum(item["compressed_history"] for item in rows),
        "precompression_targets_exactly_match_native": True,
        "parameters_unchanged": True,
        "temporal_state_restored": True,
        "attention_layer_calls": len(trace),
        "source_hashes_unchanged": True,
        "scope": "engineering gate only; scores are not an efficacy result",
    })


def orchestrate(args):
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", json.loads(args.gate_protocol.read_text()))
    sources_before = source_receipt()
    write(output / "progress.json", {"status": "running", "phase": "fresh_process_replays"})
    trials = {}
    for condition in CONDITIONS:
        trials[condition] = []
        for trial_no in (1, 2):
            trial_dir = output / f"{condition}_trial{trial_no:02d}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--checkpoint", str(args.checkpoint),
                "--output-dir", str(trial_dir),
                "--condition", condition,
                "--matching-mode", args.matching_mode,
                "--gate-protocol", str(args.gate_protocol),
                "--trial",
            ]
            env = os.environ.copy()
            env.update({"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            completion = json.loads((trial_dir / "completion.json").read_text())
            if completion.get("passed") is not True:
                raise RuntimeError(f"Engineering replay failed: {condition}/{trial_no}")
            trials[condition].append({
                "trial": trial_no,
                "setup_sha256": digest(trial_dir / "setup.json"),
                "frames_sha256": digest(trial_dir / "frames.json"),
                "attention_trace_sha256": digest(trial_dir / "attention_trace.json"),
                "provenance_sha256": digest(trial_dir / "first_compressed_target_provenance.json"),
                "completion": completion,
            })
        pair = trials[condition]
        if any(pair[0][key] != pair[1][key] for key in (
            "setup_sha256", "frames_sha256", "attention_trace_sha256", "provenance_sha256"
        )):
            raise RuntimeError(f"Fresh-process replay is not deterministic for {condition}")

    if source_receipt() != sources_before:
        raise RuntimeError("Frozen source changed during gate/replay")

    traces = {}
    for condition in CONDITIONS:
        run = output / f"{condition}_trial01"
        traces[condition] = json.loads((run / "attention_trace.json").read_text())
    for condition, other in (("position_mean", "position_representative"),):
        first_by_target = {frame: [row for row in traces[condition] if row["target_frame"] == frame]
                           for frame in {row["target_frame"] for row in traces[condition]}}
        other_by_target = {frame: [row for row in traces[other] if row["target_frame"] == frame]
                           for frame in {row["target_frame"] for row in traces[other]}}
        if first_by_target.keys() != other_by_target.keys():
            raise RuntimeError("Position modes did not process identical target frames")
        for frame in first_by_target:
            a, b = first_by_target[frame], other_by_target[frame]
            if len(a) != 4 or len(b) != 4:
                raise RuntimeError("Position modes do not cover all attention layers")
            for a_read, b_read in zip(a, b, strict=True):
                quotas_a = [(x["frame"], x["t_pos"], x["output_spatial_keys"], x["quota_sha256"])
                            for x in a_read["frames"]]
                quotas_b = [(x["frame"], x["t_pos"], x["output_spatial_keys"], x["quota_sha256"])
                            for x in b_read["frames"]]
                if quotas_a != quotas_b or a_read["pointer_keys"] != b_read["pointer_keys"]:
                    raise RuntimeError(f"Position modes used unmatched budgets at frame {frame}")

    gate = {
        "passed": True,
        "classification": "real-clip engineering validation; no efficacy claim",
        "sequence": SEQUENCE,
        "fresh_process_replays_per_condition": 2,
        "conditions": trials,
        "position_modes_used_identical_frame_and_window_budgets": True,
        "precompression_native_parity_passed": True,
        "compressed_path_exercised": True,
        "parameters_and_temporal_state_restored": True,
        "sources": sources_before,
        "completed_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write(output / "gate.json", gate)
    write(output / "progress.json", {
        "status": "complete", "passed": True,
        "conditions": list(CONDITIONS), "fresh_process_trials": 4,
    })
    print(f"PASS: representative-position long-history gate: {output / 'gate.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), default="position_representative")
    parser.add_argument("--matching-mode", choices=("key", "key_value"), default="key")
    parser.add_argument("--gate-protocol", type=Path, default=GATE_PROTOCOL)
    parser.add_argument("--trial", action="store_true")
    args = parser.parse_args()
    if args.trial:
        trial(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
