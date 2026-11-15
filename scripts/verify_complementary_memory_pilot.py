"""Seq2/seq3 engineering gate for the frozen complementary-memory pilot."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling.complementary_memory import ComplementaryMemoryReader, NativeComplementaryRead
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import tensor_hash
from scripts.run_complementary_memory_pilot import (
    MODES,
    PROTOCOL_REL,
    atomic_json,
    init_predictor,
    make_setup,
    prepare,
    save_source_snapshot,
    source_hashes,
    state_digest,
    train_video,
)
from datasets.video import load_manifest


def stored_output(bank, frame):
    output = bank["cond_frame_outputs"].get(frame)
    if output is None:
        output = bank["non_cond_frame_outputs"].get(frame)
    if output is None:
        raise RuntimeError(f"Missing stored output at frame {frame}")
    return output


def rollout(checkpoint, repo, frames, output, reader=None, mode="regional", max_targets=None):
    model = init_predictor(checkpoint, repo)
    model_hash = parameter_digest(model)
    rows, calls = [], []
    with tempfile.TemporaryDirectory(prefix="gate_rollout_", dir=output) as temporary:
        with torch.inference_mode():
            state, prompt = prepare(model, frames, Path(temporary))
        observer = NativeComplementaryRead(
            model, reader, mode, inference_state=state,
            trace_only=reader is None,
            record_update_norm=reader is not None,
        )
        iterator = None
        target_count = 0
        try:
            with torch.inference_mode():
                context = observer if observer is not None else _NullContext()
                with context:
                    iterator = model.propagate_in_video(state, start_frame_idx=prompt)
                    for frame, ids, logits in iterator:
                        if ids != [1] or not torch.isfinite(logits).all():
                            raise RuntimeError("Invalid output IDs or logits")
                        bank = state["output_dict_per_obj"][0]
                        stored = stored_output(bank, frame)
                        row = {
                            "frame": frame,
                            "prediction_sha256": tensor_hash(logits),
                            "stored_output_sha256": fingerprint(stored)["sha256"],
                            "pointer_sha256": tensor_hash(stored["obj_ptr"]),
                        }
                        if frame != prompt:
                            row["read"] = observer.calls[-1]
                            calls.append(observer.calls[-1])
                        rows.append(row)
                        if frame != prompt:
                            target_count += 1
                        if max_targets is not None and target_count >= max_targets:
                            break
        finally:
            if iterator is not None:
                iterator.close()
            with torch.inference_mode():
                model.reset_state(state)
    if parameter_digest(model) != model_hash:
        raise RuntimeError("Original SAM2 parameters changed during gate rollout")
    del model
    gc.collect()
    return {"prompt_frame": prompt, "rows": rows, "reader_calls": calls,
            "model_parameter_sha256": model_hash}


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def compare_exact(native, adapted, name, mode):
    left, right = native["rows"], adapted["rows"]
    if len(left) != len(right) or native["prompt_frame"] != adapted["prompt_frame"]:
        raise RuntimeError(f"Frame coverage differs for {name}/{mode}")
    for a, b in zip(left, right):
        for key in ("frame", "prediction_sha256", "stored_output_sha256", "pointer_sha256"):
            if a[key] != b[key]:
                raise RuntimeError(f"Zero-residual native parity failed: {name}/{mode}/{a['frame']}/{key}")
    native_calls, adapted_calls = native["reader_calls"], adapted["reader_calls"]
    if len(native_calls) != len(adapted_calls):
        raise RuntimeError(f"Native pointer trace coverage differs for {name}/{mode}")
    for base, test in zip(native_calls, adapted_calls):
        for key in ("frame", "memory_frames", "spatial_keys", "pointer_keys",
                    "pointer_input_sha256", "pointer_position_sha256"):
            if base.get(key) != test.get(key):
                raise RuntimeError(f"Native memory/pointer input changed: {name}/{mode}/{base['frame']}/{key}")
        if test["pointer_keys"] <= 0:
            raise RuntimeError(f"Expected native pointer tokens in the memory read: {name}/{test['frame']}")
    if len(adapted["reader_calls"]) != len(right) - 1:
        raise RuntimeError(f"Not all post-prompt reads were observed for {name}/{mode}")
    if any(row["layers"] != [0, 1, 2, 3] or not row["stored_mask_alignment"]
           or not row["native_positions_verified"] for row in adapted["reader_calls"]):
        raise RuntimeError(f"Incomplete memory-attention alignment for {name}/{mode}")
    return {"sequence": name, "mode": mode, "frames_including_prompt": len(right),
            "post_prompt_reads": len(adapted["reader_calls"]),
            "zero_residual_predictions_memory_and_pointers_bit_identical": True,
            "all_four_layers_and_native_positions_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    try:
        manifest = ROOT / "data/polypgen_sequence.jsonl"
        videos = load_manifest(manifest, "dev") | load_manifest(manifest, "test")
        protocol, setup_core, setup_hash, repo = make_setup(args, videos)
        if digest(ROOT / PROTOCOL_REL) != setup_core["protocol_sha256"]:
            raise RuntimeError("Protocol changed during gate setup")
        atomic_json(output / "setup.json", {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "engineering gate only; no efficacy result",
            "command": sys.argv,
            "setup_core_sha256": setup_hash,
            "setup_core": setup_core,
            "source_snapshot_sha256": save_source_snapshot(output),
        })
        native_rows, identities, training_rows, active_rows = {}, [], {}, {}
        for name in ("seq2", "seq3"):
            atomic_json(output / "status.json", {"phase": "native_parity", "sequence": name})
            native = rollout(args.checkpoint, repo, videos[name], output)
            native_rows[name] = native
            atomic_json(output / f"{name}_native.json", native)
            for mode in MODES:
                torch.manual_seed(20261106)
                reader = ComplementaryMemoryReader().eval()
                with torch.no_grad():
                    reader.output_projection.weight.zero_()
                adapted = rollout(args.checkpoint, repo, videos[name], output, reader, mode)
                row = compare_exact(native, adapted, name, mode)
                identities.append(row)
                atomic_json(output / f"{name}_{mode}_zero_residual.json", {**row, "trace": adapted["reader_calls"]})

        initial_states = {}
        for mode in MODES:
            torch.manual_seed(20261106)
            reader = ComplementaryMemoryReader().train()
            initial_states[mode] = state_digest(reader.state_dict())
        if initial_states["generic"] != initial_states["regional"]:
            raise RuntimeError("Matched arms did not start from identical parameters")

        for mode in MODES:
            torch.manual_seed(20261106)
            reader = ComplementaryMemoryReader().train()
            optimizer = torch.optim.AdamW(reader.parameters(), lr=1e-4, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=0.01)
            model = init_predictor(args.checkpoint, repo)
            model_hash = parameter_digest(model)
            rows = []
            for name in ("seq2", "seq3"):
                row = train_video(model, reader, mode, optimizer, videos[name], name,
                                  output, output / "status.json", 20261106, mode,
                                  max_post_frames=8)
                rows.append(row)
            if parameter_digest(model) != model_hash:
                raise RuntimeError("SAM2 weights changed during autoregressive training gate")
            module_gradients = {}
            for group in ("output_projection", "summary_input_projection", "feature_pool",
                          "pool_weights", "position_projection", "reader_attention"):
                module_gradients[group] = sum(
                    chunk["gradient_groups_l1"][group]
                    for sequence in rows for chunk in sequence["chunks"]
                )
            if any(value <= 0 for value in module_gradients.values()):
                raise RuntimeError(f"A required reader module received no gradient: {module_gradients}")
            if sum(row["post_prompt_frames"] for row in rows) != 16:
                raise RuntimeError("Engineering training gate did not consume the fixed 16-frame subset")
            trained_path = output / f"{mode}_reader_gate.pt"
            from scripts.run_complementary_memory_pilot import atomic_torch_save
            atomic_torch_save(trained_path, reader.state_dict())
            training_rows[mode] = {
                "sequences": rows,
                "updates": sum(row["optimizer_updates"] for row in rows),
                "frames": sum(row["post_prompt_frames"] for row in rows),
                "module_gradient_l1": module_gradients,
                "initial_reader_state_sha256": initial_states[mode],
                "trained_reader_state_sha256": state_digest(reader.state_dict()),
                "reader_checkpoint_sha256": digest(trained_path),
                "sam2_parameter_sha256": model_hash,
            }
            reader.eval()
            for name in ("seq2", "seq3"):
                active = rollout(args.checkpoint, repo, videos[name], output, reader, mode,
                                 max_targets=8)
                baseline = native_rows[name]
                expected = {row["frame"]: row["prediction_sha256"] for row in baseline["rows"]}
                changed = [row["frame"] for row in active["rows"]
                           if row["frame"] != active["prompt_frame"]
                           and row["prediction_sha256"] != expected.get(row["frame"])]
                calls = active["reader_calls"]
                if (len(calls) != 8 or any(not call.get("finite_reader_update")
                                           or call.get("reader_update_l1", 0.0) <= 0
                                           for call in calls)):
                    raise RuntimeError(f"Trained {mode} reader produced no finite residual on {name}")
                active_rows[f"{mode}/{name}"] = {
                    "scored_predictions": len(active["rows"]) - 1,
                    "changed_logits_frames": changed,
                    "prediction_masks_changed": bool(changed),
                    "sum_reader_update_l1": sum(call["reader_update_l1"] for call in calls),
                    "all_reads_four_layers": all(call["layers"] == [0, 1, 2, 3] for call in active["reader_calls"]),
                    "finite": True,
                }
            del model
            gc.collect()

        if source_hashes() != setup_core["sources"]:
            raise RuntimeError("Source files changed during the engineering gate")
        result = {
            "passed": True,
            "scope": "engineering gate only; no Dice efficacy measured",
            "sequences": identities,
            "training": training_rows,
            "active_short_rollouts": active_rows,
            "matched_initialization": initial_states,
            "source_hashes_unchanged": True,
            "elapsed_seconds": time.monotonic() - started,
            "full_cohort_training_authorized": True,
            "interpretation": "The seq2/seq3 implementation gate passed; this validates the runner and causal predicted-memory loop, not segmentation improvement.",
        }
        atomic_json(output / "gate_results.json", result)
        atomic_json(output / "completion.json", {
            "passed": True,
            "exit_code": 0,
            "setup_core_sha256": setup_hash,
            "gate_results_sha256": digest(output / "gate_results.json"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(output / "status.json", {"phase": "gate_complete", "passed": True})
    except BaseException as error:
        atomic_json(output / "failure.json", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": repr(error),
        })
        atomic_json(output / "status.json", {"phase": "gate_failed"})
        raise


if __name__ == "__main__":
    main()
