"""Full-sequence engineering gate using the repository's existing VOS runner."""

import argparse
import json
from pathlib import Path
import sys
from importlib.metadata import PackageNotFoundError, version

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.spatiotemporal_tome_value_exchange import NativeSpatiotemporalValueExchange
from scripts.memory_token_io import digest, validate_output
from scripts.run_region_constrained_tome_gate import native_reference
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import summarize
from scripts.run_wider_history_tome_gate import (
    CHECKPOINT_DEFAULT,
    NATIVE_PROTOCOL,
    NATIVE_RESULTS,
    NATIVE_SETUP,
    SAM2_REFERENCE,
    run_condition,
    verify_native_setup,
)

CONDITIONS = ("noop", "local", "cross_frame")
SEQUENCES = ("seq6", "seq8")
MANIFEST = ROOT / "data/polypgen_sequence.jsonl"


class RecordingValueExchange(NativeSpatiotemporalValueExchange):
    def __init__(self, model, *, mode, audit_sink):
        super().__init__(model, mode=mode)
        self.audit_sink = audit_sink

    def __exit__(self, *exc):
        self.audit_sink.extend(self.calls)
        return super().__exit__(*exc)


def verify_fullprop_audits(audits, target_frames):
    expected_calls = len(target_frames) * 4
    for mode in CONDITIONS:
        calls = audits[mode]
        if len(calls) != expected_calls:
            raise RuntimeError(f"{mode} logged {len(calls)} attention calls, expected {expected_calls}")
        by_frame = {}
        for call in calls:
            if not call["key_unchanged"]:
                raise RuntimeError(f"{mode} changed memory K at frame {call['frame']}")
            by_frame.setdefault(call["frame"], []).append(call)
        if list(by_frame) != list(target_frames):
            raise RuntimeError(f"{mode} read targets differ from the full sequence")
        if any(sorted(call["layer"] for call in rows) != [0, 1, 2, 3] for rows in by_frame.values()):
            raise RuntimeError(f"{mode} did not call all four layers once per target")

    cross_groups = 0
    for frame in target_frames:
        for layer in range(4):
            noop = next(x for x in audits["noop"] if x["frame"] == frame and x["layer"] == layer)
            local = next(x for x in audits["local"] if x["frame"] == frame and x["layer"] == layer)
            cross = next(x for x in audits["cross_frame"] if x["frame"] == frame and x["layer"] == layer)
            layout_fields = (
                "eligible_frames", "eligible_t_pos", "frames", "spatial_keys_before",
                "spatial_keys_after", "spatial_key_mass_sum", "key_mass_sum", "pointer_keys",
            )
            for field in layout_fields:
                if local[field] != cross[field] or local[field] != noop[field]:
                    raise RuntimeError(f"Memory layout differs across arms at {frame}/{layer}: {field}")
            if noop["scope"] != "noop":
                raise RuntimeError(f"No-op arm applied a transform at {frame}/{layer}")
            if noop["merged_pairs"] != 0 or noop["cross_frame_groups"] != 0:
                raise RuntimeError(f"No-op arm recorded ToMe merges at {frame}/{layer}")
            expected_local_scope = "local" if local["eligible_t_pos"] else "noop"
            if local["scope"] != expected_local_scope:
                raise RuntimeError(f"Local control used the wrong scope at {frame}/{layer}")
            eligible_tokens = sum(
                int(row["source_tokens"])
                for row in local["frames"]
                if int(row["t_pos"]) > 0
            )
            if (
                local["eligible_tokens"] != eligible_tokens
                or cross["eligible_tokens"] != eligible_tokens
            ):
                raise RuntimeError(f"Eligible-token audit differs at {frame}/{layer}")
            if local["cross_frame_groups"] != 0:
                raise RuntimeError(f"Local control formed a cross-frame match at {frame}/{layer}")
            if local["merged_pairs"] != cross["merged_pairs"]:
                raise RuntimeError(f"Matched merge counts differ at {frame}/{layer}")
            if local["merged_pairs"] != local["eligible_tokens"] // 4:
                raise RuntimeError(f"Matched merge budget differs at {frame}/{layer}")
            expected_scope = (
                "cross_frame" if len(cross["eligible_t_pos"]) > 1
                else "local" if cross["eligible_t_pos"]
                else "noop"
            )
            if cross["scope"] != expected_scope:
                raise RuntimeError(f"Cross-frame mode used the wrong scope at {frame}/{layer}")
            if expected_scope == "cross_frame" and cross["cross_frame_groups"] <= 0:
                raise RuntimeError(
                    f"No cross-frame ToMe group at target {frame}, layer {layer}"
                )
            cross_groups += cross["cross_frame_groups"]

    if cross_groups <= 0:
        raise RuntimeError("Full-sequence propagation formed no cross-frame ToMe groups")
    return {
        "attention_calls_per_arm": expected_calls,
        "cross_frame_groups": cross_groups,
        "local_cross_frame_groups": 0,
        "layout_and_budget_match": True,
        "native_keys_unchanged": True,
    }


def count_fullprop_outputs(rows_by_sequence):
    """Count unique target frames separately from per-condition predictions."""
    sequence_targets = {}
    condition_frame_outputs = 0
    for sequence, arms in rows_by_sequence.items():
        if set(arms) != set(CONDITIONS):
            raise RuntimeError(f"Incomplete conditions for completion counts: {sequence}")
        counts = {mode: len(arms[mode]) for mode in CONDITIONS}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"Condition frame counts differ for completion: {sequence}")
        sequence_targets[sequence] = counts[CONDITIONS[0]]
        condition_frame_outputs += sum(counts.values())
    return {
        "sequence_targets": sequence_targets,
        "unique_post_prompt_targets": sum(sequence_targets.values()),
        "condition_frame_outputs": condition_frame_outputs,
    }


def _load_protocol(path):
    protocol = json.loads(path.read_text())
    checksum = path.with_suffix(path.suffix + ".sha256")
    if protocol.get("status") != "frozen_for_fullprop_engineering_gate":
        raise RuntimeError("Full-propagation protocol is not frozen")
    if digest(path) != checksum.read_text().strip():
        raise RuntimeError("Full-propagation protocol checksum differs")
    for relative, expected in protocol["source_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen source changed: {relative}")
    for name, expected in protocol["input_hashes"].items():
        if digest(Path(name)) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    if len(protocol["gate_runs"]) != 2:
        raise RuntimeError("Require two prior cached-read gate receipts")
    gate_results = []
    for attempt in protocol["gate_runs"]:
        folder = Path(attempt["directory"])
        setup = json.loads((folder / "setup.json").read_text())
        completion = json.loads((folder / "completion.json").read_text())
        result_path = folder / "results.json"
        results_sha256 = digest(result_path)
        if (
            setup.get("protocol_sha256") != attempt["protocol_sha256"]
            or not completion.get("passed")
            or completion.get("result_sha256") != results_sha256
            or results_sha256 != attempt["results_sha256"]
        ):
            raise RuntimeError(f"Prior read gate is incomplete or changed: {folder}")
        gate_results.append(result_path.read_bytes())
    if gate_results[0] != gate_results[1]:
        raise RuntimeError("Prior two-process read gate outputs are not byte-identical")
    return protocol


def _preflight_metric_dependency(expected_version):
    try:
        from skimage.morphology import disk
        actual_version = version("scikit-image")
    except (ImportError, PackageNotFoundError) as exc:
        raise RuntimeError(
            "The existing DAVIS boundary metric requires scikit-image; install the "
            "protocol-pinned version before starting inference"
        ) from exc
    if actual_version != expected_version:
        raise RuntimeError(
            f"scikit-image version {actual_version} differs from frozen "
            f"version {expected_version}"
        )
    if disk(1).shape != (3, 3):
        raise RuntimeError("scikit-image morphology disk preflight returned an unexpected kernel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load_protocol(args.protocol)
    runtime = protocol["runtime"]
    if runtime.get("bytecode_writes_disabled") and not sys.dont_write_bytecode:
        raise RuntimeError("Run with PYTHONDONTWRITEBYTECODE=1 as frozen")
    if Path(sys.executable).resolve() != Path(runtime["runner"]).resolve():
        raise RuntimeError("Python executable differs from the frozen gate environment")
    if f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}" != runtime["python"]:
        raise RuntimeError("Python version differs from the frozen gate environment")
    if torch.__version__.split("+", 1)[0] != runtime["torch"]:
        raise RuntimeError("PyTorch version differs from the frozen gate environment")
    _preflight_metric_dependency(runtime["scikit_image"])
    output = validate_output(args.output_dir)
    output_root = Path(protocol["output_root"]).resolve()
    if not output.is_relative_to(output_root) or output.exists():
        raise RuntimeError("Choose a new child folder under the frozen Drive output root")
    output.mkdir(parents=True)
    write(output / "setup.json", {"protocol": protocol, "protocol_sha256": digest(args.protocol)})

    torch.set_num_threads(runtime["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(runtime["seed"])
    native_setup = verify_native_setup(CHECKPOINT_DEFAULT, MANIFEST)
    if digest(CHECKPOINT_DEFAULT) != native_setup["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from the verified native setup")
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    native_data = json.loads(NATIVE_RESULTS.read_text())
    videos = {**load_manifest(MANIFEST, "dev"), **load_manifest(MANIFEST, "test")}

    rows_by_sequence, summaries, runtimes, attention_audits = {}, {}, {}, {}
    try:
        for sequence in protocol["sequences"]:
            frames = videos[sequence]
            prompt, native_rows, receipt = native_reference(
                NATIVE_RESULTS.parent, native_data, native_protocol, sequence, frames
            )
            if len(frames) - prompt - 1 != len(native_rows):
                raise RuntimeError(f"Native reference has incomplete coverage for {sequence}")
            target_frames = list(range(prompt + 1, len(frames)))
            rows_by_sequence[sequence], summaries[sequence], runtimes[sequence] = {}, {}, {}
            attention_audits[sequence] = {}

            for mode in CONDITIONS:
                write(output / "progress.json", {
                    "status": "running", "sequence": sequence, "mode": mode,
                    "targets_total": len(target_frames),
                })
                raw_audit = []
                result_rows, run_receipt = run_condition(
                    "global", sequence, frames, prompt, native_rows,
                    CHECKPOINT_DEFAULT, output,
                    max_noncond_maps=6,
                    allocation="uniform",
                    condition_name=mode,
                    memory_reader_factory=lambda model, chosen=mode, sink=raw_audit: RecordingValueExchange(
                        model, mode=chosen, audit_sink=sink
                    ),
                    native_exact_required=(mode == "noop"),
                )
                if [row["frame"] for row in result_rows] != target_frames:
                    raise RuntimeError(f"Frame coverage differs for {sequence}/{mode}")
                if any(row["present"] != native_rows[row["frame"]]["present"] for row in result_rows):
                    raise RuntimeError(f"Ground-truth frame presence differs for {sequence}/{mode}")
                if mode == "noop":
                    for row in result_rows:
                        reference = native_rows[row["frame"]]
                        for field in ("prediction_sha256", "stored_sha256", "scores", "absent_fp"):
                            if row[field] != reference[field]:
                                raise RuntimeError(f"Native/no-op mismatch: {sequence}/{row['frame']}/{field}")

                rows_by_sequence[sequence][mode] = result_rows
                summaries[sequence][mode] = summarize(result_rows)
                runtimes[sequence][mode] = run_receipt
                attention_audits[sequence][mode] = raw_audit

            attention_audits[sequence] = verify_fullprop_audits(
                attention_audits[sequence], target_frames
            )

        result = {
            "scope": "two-sequence full-propagation engineering diagnostic; not cohort efficacy",
            "sequences": list(protocol["sequences"]),
            "conditions": list(CONDITIONS),
            "sequence_summaries": summaries,
            "sequence_rows": rows_by_sequence,
            "attention_audit": attention_audits,
            "native_noop_exact": True,
        }
        for relative, expected in protocol["source_hashes"].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f"Source changed during propagation: {relative}")
        write(output / "results.json", result)
        write(output / "runtime.json", runtimes)
        output_counts = count_fullprop_outputs(rows_by_sequence)
        write(output / "completion.json", {
            "passed": True,
            "scope": result["scope"],
            "sequences": list(protocol["sequences"]),
            **output_counts,
            "results_sha256": digest(output / "results.json"),
        })
        write(output / "progress.json", {"status": "complete"})
    except BaseException as exc:
        write(output / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        write(output / "progress.json", {
            "status": "failed",
            "sequence": sequence if "sequence" in locals() else None,
            "mode": mode if "mode" in locals() else None,
            "error": str(exc),
        })
        raise


if __name__ == "__main__":
    main()
