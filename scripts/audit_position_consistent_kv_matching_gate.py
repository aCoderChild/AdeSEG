"""Audit fresh-process replay and matched budgets for the K versus K/V gate."""

import argparse
import hashlib
import json
from pathlib import Path


def _read(path):
    return json.loads(path.read_text())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trial(root, name, expected_mode):
    path = root / name
    completion = _read(path / "completion.json")
    setup = _read(path / "setup.json")
    if completion.get("passed") is not True:
        raise RuntimeError(f"{name} did not pass its real-clip gate")
    if completion.get("matching_mode") != expected_mode:
        raise RuntimeError(f"{name} used the wrong matching metric")
    if setup.get("matching_mode") != expected_mode:
        raise RuntimeError(f"{name} setup does not record the expected metric")
    if setup.get("position_mode") != "representative":
        raise RuntimeError(f"{name} did not hold position handling fixed")
    if completion.get("targets") != 14 or completion.get("compressed_targets") != 7:
        raise RuntimeError(f"{name} did not cover the expected seq3 trajectory")
    if completion.get("attention_layer_calls") != 56:
        raise RuntimeError(f"{name} did not exercise all four layers per target")
    return {
        "name": name,
        "path": path,
        "setup": setup,
        "completion": completion,
        "frames": _read(path / "frames.json"),
        "trace": _read(path / "attention_trace.json"),
        "provenance": _read(path / "first_compressed_target_provenance.json"),
        "file_hashes": {
            filename: _sha(path / filename)
            for filename in (
                "setup.json",
                "completion.json",
                "frames.json",
                "attention_trace.json",
                "first_compressed_target_provenance.json",
            )
        },
    }


def _budget_signature(read):
    return (
        read["target_frame"],
        read["layer"],
        read["position_mode"],
        read["spatial_keys_before"],
        read["spatial_keys_after"],
        read["pointer_keys"],
        tuple(
            (
                frame["frame"],
                frame["t_pos"],
                frame["input_spatial_keys"],
                frame["output_spatial_keys"],
                frame["quota_sha256"],
            )
            for frame in read["frames"]
        ),
    )


def _partition_signature(window):
    return tuple(sorted(
        (
            tuple(sorted(int(index) for index in cluster["original_grid_indices"])),
            int(cluster["representative_grid_index"]),
        )
        for cluster in window["clusters"]
    ))


def audit(root):
    trials = {
        "key": [_trial(root, f"key_trial{index:02d}", "key") for index in (1, 2)],
        "key_value": [
            _trial(root, f"key_value_trial{index:02d}", "key_value")
            for index in (1, 2)
        ],
    }
    for mode, pair in trials.items():
        if pair[0]["file_hashes"] != pair[1]["file_hashes"]:
            raise RuntimeError(f"Fresh-process {mode} replays differ")

    reference = trials["key"][0]
    candidate = trials["key_value"][0]
    shared_setup_fields = (
        "checkpoint_sha256",
        "manifest_sha256",
        "native_results_sha256",
        "sam2_revision",
        "tome_revision",
        "sequence",
        "prompt",
        "target_frames",
        "inputs",
        "device",
        "dtype",
        "runtime",
        "sources",
    )
    if any(
        reference["setup"].get(field) != candidate["setup"].get(field)
        for field in shared_setup_fields
    ):
        raise RuntimeError("K-only and K/V conditions did not use the same frozen inputs")
    if len(reference["frames"]) != len(candidate["frames"]):
        raise RuntimeError("K-only and K/V conditions did not produce aligned target rows")
    if len(reference["trace"]) != 56 or len(candidate["trace"]) != 56:
        raise RuntimeError("Unexpected attention trace length")
    by_frame_key = {row["frame"]: row for row in reference["frames"]}
    by_frame_kv = {row["frame"]: row for row in candidate["frames"]}
    if by_frame_key.keys() != by_frame_kv.keys():
        raise RuntimeError("K-only and K/V target frame IDs differ")
    for frame in by_frame_key:
        key_row, kv_row = by_frame_key[frame], by_frame_kv[frame]
        if (key_row["present"], key_row["compressed_history"]) != (
            kv_row["present"], kv_row["compressed_history"]
        ):
            raise RuntimeError("K-only and K/V presence or compression labels differ")
        if not key_row["compressed_history"]:
            if not key_row.get("exact_native_prefix") or not kv_row.get("exact_native_prefix"):
                raise RuntimeError("A pre-compression target lacks exact native parity")
            if any(
                key_row.get(field) != kv_row.get(field)
                for field in ("prediction_sha256", "stored_sha256", "scores", "absent_fp")
            ):
                raise RuntimeError("K/V matching changed a pre-compression trajectory output")
    for key_read, kv_read in zip(reference["trace"], candidate["trace"], strict=True):
        if _budget_signature(key_read) != _budget_signature(kv_read):
            raise RuntimeError("K-only and K/V conditions used unmatched token budgets")

    changed_partitions = 0
    changed_predictions_after_compression = 0
    reference_prov = reference["provenance"]
    candidate_prov = candidate["provenance"]
    if len(reference_prov) != len(candidate_prov):
        raise RuntimeError("Cluster-provenance layer counts differ")
    for left, right in zip(reference_prov, candidate_prov, strict=True):
        for left_frame, right_frame in zip(left["frames"], right["frames"], strict=True):
            for left_window, right_window in zip(
                left_frame["windows"], right_frame["windows"], strict=True
            ):
                if _partition_signature(left_window) != _partition_signature(right_window):
                    changed_partitions += 1
    for frame in by_frame_key:
        if by_frame_key[frame]["compressed_history"] and (
            by_frame_key[frame]["prediction_sha256"]
            != by_frame_kv[frame]["prediction_sha256"]
        ):
            changed_predictions_after_compression += 1
        if not by_frame_key[frame]["compressed_history"] and (
            by_frame_key[frame]["prediction_sha256"]
            != by_frame_kv[frame]["prediction_sha256"]
        ):
            raise RuntimeError("K/V matching changed a pre-compression prediction")

    result = {
        "status": "engineering_integrity_passed",
        "scope": "seq3 real-clip engineering gate; no Dice efficacy inference",
        "sequence": "seq3",
        "targets_per_condition": 14,
        "compressed_targets_per_condition": 7,
        "attention_layer_calls_per_condition": 56,
        "fresh_process_replays_per_condition": 2,
        "replay_exact": True,
        "k_only_vs_kv_budget_and_input_match": True,
        "native_prefix_parity": True,
        "changed_window_partitions_at_first_compressed_target": changed_partitions,
        "compressed_targets_with_changed_prediction_hash": changed_predictions_after_compression,
        "conditions": {
            mode: [trial["file_hashes"] for trial in pair]
            for mode, pair in trials.items()
        },
    }
    (root / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps({
        key: value for key, value in result.items() if key != "conditions"
    }, indent=2))


if __name__ == "__main__":
    main()
