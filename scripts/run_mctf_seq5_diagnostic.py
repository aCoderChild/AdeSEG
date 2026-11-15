"""Exploratory Seq5 full propagation for frozen MCTF memory-value study.

This single-sequence follow-up was selected after observing prolonged native
object loss. It is diagnostic only: it is not a confirmatory efficacy test and
must not be used to tune the method or claim a cohort improvement.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from datasets.video import load_manifest
from modeling.mctf_memory_value import MCTFMemoryValue
from modeling.native_value_consolidation import NativeValueConsolidation
from scripts.memory_token_io import digest, validate_output
from scripts.run_region_constrained_tome_gate import native_reference
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import summarize
from scripts import run_wider_history_tome_gate as wider_runner
from scripts.run_wider_history_tome_gate import (
    CHECKPOINT_DEFAULT,
    NATIVE_PROTOCOL,
    NATIVE_RESULTS,
    run_condition,
    verify_native_setup,
)

PROTOCOL_DEFAULT = ROOT / "protocols/mctf_memory_value_seq5_diagnostic_v1.json"
ARMS = ("generic_tome_value", "mctf_uniform", "mctf_attention")


def _compact_mctf_read(call):
    """Serialize dense value consolidation without pretending tokens were removed."""
    frames = []
    for row in call["frames"]:
        frames.append({
            "frame": int(row["frame"]),
            "t_pos": int(row["t_pos"]),
            "source_tokens": int(row["source_tokens"]),
            "target_tokens": int(row["source_tokens"]),
            "merged_tokens": int(row["merged_tokens"]),
            "windows": int(row.get("windows", 0)),
        })
    return {
        "layer": int(call["layer"]),
        "spatial_keys_before": int(call["spatial_keys_before"]),
        "spatial_keys_after": int(call["spatial_keys_after"]),
        "pointer_keys": int(call["pointer_keys"]),
        "spatial_key_mass_sum": float(call["spatial_key_mass_sum"]),
        "key_mass_sum": float(call["key_mass_sum"]),
        "frames": frames,
    }


def _assert_mctf_integrity(read, expected_max_noncond=6):
    if read["spatial_keys_after"] != read["spatial_keys_before"]:
        raise RuntimeError("Value-only consolidation must preserve every spatial key")
    if read["spatial_key_mass_sum"] != float(read["spatial_keys_before"]):
        raise RuntimeError("Spatial key accounting differs from native")
    if read["key_mass_sum"] != float(read["spatial_keys_before"] + read["pointer_keys"]):
        raise RuntimeError("Pointer-inclusive key accounting differs from native")
    if len([item for item in read["frames"] if item["t_pos"] > 0]) > expected_max_noncond:
        raise RuntimeError("Seq5 diagnostic exceeded native non-conditioning history")
    if sum(item["target_tokens"] for item in read["frames"]) != read["spatial_keys_before"]:
        raise RuntimeError("Dense value slots do not cover the native spatial key grid")


class _Seq5MCTFReader(MCTFMemoryValue):
    """MCTF reader with native-dense accounting expected by the shared runner."""

    def _sdpa(self, query, key, value, *args, **kwargs):
        output = super()._sdpa(query, key, value, *args, **kwargs)
        if self.active is None or self.inside:
            return output
        call = self.calls[-1]
        pointers = int(call["pointer_keys"])
        spatial = int(key.shape[-2] - pointers)
        call.update({
            "spatial_keys_before": spatial,
            "spatial_keys_after": spatial,
            "spatial_key_mass_sum": float(spatial),
            "key_mass_sum": float(key.shape[-2]),
        })
        by_frame = {
            int(span["frame"]): {
                "tokens": int(span["stop"] - span["start"]),
                "t_pos": int(span["t_pos"]),
            }
            for span in self.spans
        }
        for item in call["frames"]:
            span = by_frame[int(item["frame"])]
            item["source_tokens"] = span["tokens"]
            item["t_pos"] = span["t_pos"]
        return output


# Reuse the established full-video propagation harness, adapting only its
# recorder to this method's dense-slot semantics. The inference path is intact.
wider_runner._compact_read = _compact_mctf_read
wider_runner._assert_read_integrity = _assert_mctf_integrity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    checksum = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    if protocol.get("status") != "frozen_for_seq5_diagnostic":
        raise RuntimeError("Seq5 protocol is not frozen")
    if digest(args.protocol) != checksum.read_text().strip():
        raise RuntimeError("Seq5 protocol hash mismatch")
    out = validate_output(args.output_dir)
    if not out.is_relative_to(Path(protocol["output_root"]).resolve()) or out.exists():
        raise RuntimeError("Use a new Drive child directory under the frozen output root")
    for source, expected in protocol["source_hashes"].items():
        if digest(ROOT / source) != expected:
            raise RuntimeError(f"Frozen source differs: {source}")
    for name, expected in protocol["input_hashes"].items():
        if digest(Path(name)) != expected:
            raise RuntimeError(f"Frozen input differs: {name}")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(protocol["seed"])
    manifest = ROOT / "data/polypgen_sequence.jsonl"
    checkpoint = Path(protocol["native_checkpoint"])
    verify_native_setup(checkpoint, manifest)
    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    frames = videos["seq5"]
    native_results = json.loads(NATIVE_RESULTS.read_text())
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    prompt, native_rows, native_receipt = native_reference(
        NATIVE_RESULTS.parent, native_results, native_protocol, "seq5", frames
    )
    if prompt != protocol["prompt"] or len(frames) != protocol["num_frames"]:
        raise RuntimeError("Seq5 prompt or video length differs from frozen protocol")

    out.mkdir(parents=True)
    write(out / "setup.json", {
        "protocol": protocol,
        "protocol_sha256": digest(args.protocol),
        "native_reference": native_receipt,
        "scope": "post-hoc exploratory Seq5 full propagation; no cohort inference",
    })
    native_list = list(native_rows.values())
    results = {"native": {"summary": summarize(native_list), "runtime": None}}
    factories = {
        "generic_tome_value": lambda model: NativeValueConsolidation(model, mode="value"),
        "mctf_uniform": lambda model: _Seq5MCTFReader(model, importance=False),
        "mctf_attention": lambda model: _Seq5MCTFReader(model, importance=True),
    }
    try:
        reuse = protocol.get("reuse_generic_reference")
        if reuse:
            reuse_dir = Path(reuse["directory"])
            for relative, expected in reuse["files_sha256"].items():
                if digest(reuse_dir / relative) != expected:
                    raise RuntimeError(f"Archived generic comparator changed: {relative}")
            archived_setup = json.loads((reuse_dir / "setup.json").read_text())
            archived_protocol = archived_setup["protocol"]
            if archived_setup["protocol_sha256"] != reuse["protocol_sha256"]:
                raise RuntimeError("Archived generic setup protocol hash differs")
            old_code_hash = archived_protocol["source_hashes"].get("scripts/run_mctf_seq5_diagnostic.py")
            if old_code_hash != reuse["runner_sha256"]:
                raise RuntimeError("Archived generic runner hash differs from its frozen setup")
            archived = json.loads((reuse_dir / "partial_results.json").read_text())
            generic_rows = archived.get("generic_tome_value", {}).get("frames", [])
            expected_frames = list(range(prompt + 1, len(frames)))
            if [row["frame"] for row in generic_rows] != expected_frames:
                raise RuntimeError("Archived generic comparator has incomplete frame coverage")
            if summarize(generic_rows) != archived["generic_tome_value"]["summary"]:
                raise RuntimeError("Archived generic comparator summary does not recompute")
            archived_native = archived.get("native", {}).get("summary")
            if archived_native != results["native"]["summary"]:
                raise RuntimeError("Archived generic run used a different native summary")
            generic_runtime = archived["generic_tome_value"]["runtime"]
            if (generic_runtime["targets"] != len(generic_rows)
                    or generic_runtime["attention_layer_calls"] != 4 * len(generic_rows)
                    or not generic_runtime["parameter_hash_unchanged"]
                    or not generic_runtime["temporal_slots_restored"]):
                raise RuntimeError("Archived generic comparator lacks full-run integrity receipt")
            results["generic_tome_value"] = archived["generic_tome_value"]
            write(out / "reused_generic_receipt.json", {
                "directory": str(reuse_dir),
                "protocol_sha256": reuse["protocol_sha256"],
                "runner_sha256": reuse["runner_sha256"],
                "rows": len(generic_rows),
                "summary_recomputed": True,
                "runtime_integrity_checks": generic_runtime,
                "files_sha256": reuse["files_sha256"],
                "interpretation": "Completed generic-control branch reused from a process whose subsequent MCTF hook failed before inference; comparator itself completed all frame audits.",
            })
        else:
            write(out / "progress.json", {
                "status": "running", "sequence": "seq5", "arm": "generic_tome_value",
                "completed_arms": list(results), "total_arms_including_native": 4,
            })
            rows, runtime = run_condition(
                "global", "seq5", frames, prompt, native_rows, checkpoint, out,
                max_noncond_maps=6,
                allocation="uniform",
                condition_name="generic_tome_value",
                memory_reader_factory=factories["generic_tome_value"],
                native_exact_required=False,
            )
            results["generic_tome_value"] = {"summary": summarize(rows), "runtime": runtime, "frames": rows}
            write(out / "partial_results.json", results)

        for arm in ("mctf_uniform", "mctf_attention"):
            write(out / "progress.json", {
                "status": "running", "sequence": "seq5", "arm": arm,
                "completed_arms": list(results), "total_arms_including_native": 4,
            })
            rows, runtime = run_condition(
                "global", "seq5", frames, prompt, native_rows, checkpoint, out,
                max_noncond_maps=6,
                allocation="uniform",
                condition_name=arm,
                memory_reader_factory=factories[arm],
                native_exact_required=False,
            )
            if len(rows) != len(native_rows) or [x["frame"] for x in rows] != list(native_rows):
                raise RuntimeError(f"{arm}: frame coverage differs from native")
            if any(row["present"] != native_rows[row["frame"]]["present"] for row in rows):
                raise RuntimeError(f"{arm}: GT-present frame set differs from native")
            results[arm] = {
                "summary": summarize(rows),
                "runtime": runtime,
                "frames": rows,
            }
            write(out / "partial_results.json", results)

        native = results["native"]["summary"]
        contrasts = {}
        for arm in ARMS:
            rows = results[arm]["frames"]
            contrasts[arm] = {
                "present_dice_delta_vs_native": results[arm]["summary"]["present_dice"] - native["present_dice"],
                "boundary_f_delta_vs_native": results[arm]["summary"]["boundary_f"] - native["boundary_f"],
                "zero_dice_gt_present_frames": [
                    row["frame"] for row in rows if row["present"] and row["scores"]["dice"] == 0.0
                ],
                "negative_presence_on_gt_present_zero_dice": sum(
                    row["present"] and row["scores"]["dice"] == 0.0 and row["object_score"] <= 0
                    for row in rows
                ),
            }
        result = {
            "scope": "Exploratory single-sequence follow-up selected after Seq5 failure inspection; not confirmatory and not a tuning basis.",
            "native": results["native"],
            "treatments": {arm: results[arm] for arm in ARMS},
            "contrasts": contrasts,
        }
        for source, expected in protocol["source_hashes"].items():
            if digest(ROOT / source) != expected:
                raise RuntimeError(f"Frozen source changed during run: {source}")
        write(out / "results.json", result)
        write(out / "completion.json", {
            "status": "complete",
            "sequence": "seq5",
            "treatment_arms": list(ARMS),
            "targets_per_treatment": len(native_rows),
            "scope": "exploratory only; no cohort efficacy claim",
            "results_sha256": digest(out / "results.json"),
        })
        write(out / "progress.json", {"status": "complete", "sequence": "seq5"})
    except Exception as exc:
        write(out / "failure.json", {"error": str(exc), "type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
