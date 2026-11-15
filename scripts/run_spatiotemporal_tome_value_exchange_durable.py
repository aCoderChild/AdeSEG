"""Engineering prototype for durable cohort attention-audit checkpoints.

The original frozen protocol cannot authorize this modified runner. Resume is
restricted to attempts created by this runner with the same frozen protocol.

Uses the existing SAM2 video runner, native reference, frame scorer, official
Video-ToMe merge operators, and the already-gated attention adapter. This file
only coordinates the frozen cohort and records its results.
"""

import argparse
import atexit
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from modeling.spatiotemporal_tome_value_exchange import (
    NativeSpatiotemporalValueExchange,
    author_video_tome,
)
from scripts.cohort_audit_checkpoints import (
    archive_incomplete_condition,
    load_condition_checkpoint,
    save_condition_checkpoint,
    save_sequence_checkpoint,
)
from scripts.memory_token_io import digest, validate_output
from scripts.run_region_constrained_tome_gate import native_reference
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import summarize
from scripts.run_spatiotemporal_value_exchange_fullprop_gate import (
    CONDITIONS,
    RecordingValueExchange,
    count_fullprop_outputs,
    verify_fullprop_audits,
)
from scripts.run_wider_history_tome_gate import (
    CHECKPOINT_DEFAULT,
    NATIVE_PROTOCOL,
    NATIVE_RESULTS,
    NATIVE_SETUP,
    SAM2_REFERENCE,
    activate_reference_repo,
    run_condition,
    verify_native_setup,
)

MANIFEST = ROOT / "data/polypgen_sequence.jsonl"
SEQUENCES = tuple(f"seq{i}" for i in range(2, 24) if i != 7)


def load_protocol(path):
    protocol = json.loads(path.read_text())
    if protocol.get("status") != "frozen_for_cohort_efficacy":
        raise RuntimeError("Cohort protocol is not frozen for efficacy")
    if digest(path) != path.with_suffix(path.suffix + ".sha256").read_text().strip():
        raise RuntimeError("Cohort protocol checksum differs")
    if tuple(protocol.get("sequences", ())) != SEQUENCES:
        raise RuntimeError("Protocol sequence cohort differs from the 21 eligible sequences")
    if protocol.get("conditions") != list(CONDITIONS):
        raise RuntimeError("Protocol conditions differ from the audited full-propagation gate")
    own_source = str(Path(__file__).resolve().relative_to(ROOT))
    if protocol["source_hashes"].get(own_source) != digest(Path(__file__)):
        raise RuntimeError("Durable runner is not pinned in this protocol")
    if protocol["source_hashes"].get("scripts/cohort_audit_checkpoints.py") != digest(ROOT / "scripts/cohort_audit_checkpoints.py"):
        raise RuntimeError("Checkpoint helper is not pinned in this protocol")
    for relative, expected in protocol["source_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen source changed: {relative}")
    for name, expected in protocol["input_hashes"].items():
        if digest(Path(name)) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    gate = protocol["engineering_gate"]
    for attempt in gate["attempts"]:
        folder = Path(attempt["folder"])
        setup = json.loads((folder / "setup.json").read_text())
        completion = json.loads((folder / "completion.json").read_text())
        result_path = folder / "results.json"
        if (
            setup.get("protocol_sha256") != gate["protocol_sha256"]
            or not completion.get("passed")
            or digest(folder / "completion.json") != attempt["completion_sha256"]
            or digest(result_path) != attempt["results_sha256"]
            or completion.get("results_sha256") != attempt["results_sha256"]
            or completion.get("unique_post_prompt_targets") != 138
            or completion.get("condition_frame_outputs") != 414
        ):
            raise RuntimeError(f"Full-propagation gate receipt failed: {folder}")
    audit_path = Path(gate["repeatability_audit"])
    if digest(audit_path) != gate["repeatability_audit_sha256"]:
        raise RuntimeError("Full-propagation repeatability audit changed")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "passed" or not audit.get("results_byte_identical"):
        raise RuntimeError("Full-propagation repeatability gate did not pass")
    return protocol


def preflight(protocol, checkpoint=CHECKPOINT_DEFAULT):
    if digest(MANIFEST) != protocol["manifest_sha256"]:
        raise RuntimeError("PolypGen manifest differs from the frozen cohort")
    if digest(checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError("SAM2 checkpoint differs from the frozen cohort")
    native_setup = verify_native_setup(checkpoint, MANIFEST)
    revision = activate_reference_repo()
    if revision != protocol["sam2_reference_revision"]:
        raise RuntimeError("SAM2 source revision differs from the frozen cohort")
    tome = author_video_tome()
    if not all(hasattr(tome, name) for name in ("bipartite_soft_matching", "merge_wavg")):
        raise RuntimeError("Pinned Video-ToMe operators are unavailable")

    videos = {**load_manifest(MANIFEST, "dev"), **load_manifest(MANIFEST, "test")}
    if not set(SEQUENCES).issubset(videos):
        raise RuntimeError("The manifest is missing eligible sequences")
    expected = protocol["sequence_targets"]
    if set(expected) != set(SEQUENCES):
        raise RuntimeError("Frozen target plan does not list all eligible sequences")
    native_data = json.loads(NATIVE_RESULTS.read_text())
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    target_total = 0
    for sequence in SEQUENCES:
        frames = videos[sequence]
        prompt, rows, _ = native_reference(NATIVE_RESULTS.parent, native_data, native_protocol, sequence, frames)
        target_frames = list(range(prompt + 1, len(frames)))
        if list(rows) != target_frames:
            raise RuntimeError(f"Native frame coverage differs for {sequence}")
        plan = expected[sequence]
        present_targets = sum(bool(rows[frame]["present"]) for frame in target_frames)
        gt_empty_targets = len(target_frames) - present_targets
        if (
            plan["frames"] != len(frames)
            or plan["prompt"] != prompt
            or plan["targets"] != len(target_frames)
            or plan["present_targets"] != present_targets
            or plan["gt_empty_targets"] != gt_empty_targets
        ):
            raise RuntimeError(f"Frozen native-reference plan differs for {sequence}")
        target_total += len(target_frames)
    if target_total != protocol["unique_post_prompt_targets"]:
        raise RuntimeError("Frozen unique target count differs from archived native references")

    model = protocol["runtime"]
    if Path(sys.executable).resolve() != Path(model["runner"]).resolve():
        raise RuntimeError("Python executable differs from the frozen environment")
    if torch.__version__.split("+", 1)[0] != model["torch"]:
        raise RuntimeError("PyTorch version differs from the frozen environment")
    if f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}" != model["python"]:
        raise RuntimeError("Python version differs from the frozen environment")
    if np.__version__ != model["numpy"] or scipy.__version__ != model["scipy"]:
        raise RuntimeError("NumPy or SciPy differs from the frozen statistics environment")
    try:
        from skimage.morphology import disk
        actual_skimage = version("scikit-image")
    except (ImportError, PackageNotFoundError) as exc:
        raise RuntimeError("The existing Boundary F metric requires scikit-image") from exc
    if actual_skimage != model["scikit_image"] or disk(1).shape != (3, 3):
        raise RuntimeError("scikit-image differs from the frozen metric environment")
    return {
        "passed": True,
        "sequences": len(SEQUENCES),
        "unique_post_prompt_targets": target_total,
        "condition_frame_outputs": target_total * len(CONDITIONS),
        "native_checkpoint_sha256": native_setup["checkpoint_sha256"],
        "sam2_revision": revision,
        "upstream_tome_commit": protocol["upstream_tome_commit"],
        "source_hash_count": len(protocol["source_hashes"]),
        "input_hash_count": len(protocol["input_hashes"]),
    }


def cohort_comparison(sequence_summaries, statistics):
    if set(sequence_summaries) != set(CONDITIONS):
        raise ValueError("Need summaries for noop, local and cross_frame")
    if any(set(arms) != set(SEQUENCES) for arms in sequence_summaries.values()):
        raise ValueError("Every condition must cover the frozen 21-sequence cohort")

    comparisons = {}
    definitions = (
        ("cross_frame_minus_noop", "cross_frame", "noop"),
        ("cross_frame_minus_local", "cross_frame", "local"),
    )
    for name, treatment, control in definitions:
        deltas = {
            seq: sequence_summaries[treatment][seq]["present_dice"]
            - sequence_summaries[control][seq]["present_dice"]
            for seq in SEQUENCES
        }
        left = {seq: sequence_summaries[treatment][seq]["present_dice"] for seq in SEQUENCES}
        right = {seq: sequence_summaries[control][seq]["present_dice"] for seq in SEQUENCES}
        wins = sum(value > 1e-12 for value in deltas.values())
        ties = sum(abs(value) <= 1e-12 for value in deltas.values())
        comparisons[name] = {
            **paired_wilcoxon(left, right),
            "mean_delta": float(np.mean(list(deltas.values()))),
            "ci95": sequence_bootstrap_interval(
                deltas,
                seed=statistics["bootstrap_seed"],
                resamples=statistics["bootstrap_resamples"],
                confidence=statistics["confidence"],
            ),
            "win_tie_loss": [wins, ties, len(deltas) - wins - ties],
            "mean_delta_without_seq8": float(np.mean([v for s, v in deltas.items() if s != "seq8"])),
            "deltas": deltas,
        }
    adjusted = holm_adjust({name: row["p"] for name, row in comparisons.items()})
    for name, row in comparisons.items():
        row["holm_p"] = adjusted[name]

    cohort = {}
    for condition, arms in sequence_summaries.items():
        cohort[condition] = {
            metric: float(np.mean([arms[seq][metric] for seq in SEQUENCES]))
            for metric in ("present_dice", "iou", "boundary_f")
        }
        for metric in ("gt_empty_false_alarm_count", "gt_empty_targets", "return_count", "undetected_returns"):
            cohort[condition][metric] = sum(int(arms[seq][metric]) for seq in SEQUENCES)
    native, local, cross = (cohort[name] for name in ("noop", "local", "cross_frame"))
    primary = comparisons["cross_frame_minus_noop"]
    control = comparisons["cross_frame_minus_local"]
    screen = {
        "gain_vs_native_at_least_0_01": primary["mean_delta"] >= 0.01,
        "gain_vs_local_at_least_0_002": control["mean_delta"] >= 0.002,
        "both_ci_lower_positive": primary["ci95"][0] > 0 and control["ci95"][0] > 0,
        "both_holm_p_below_0_05": primary["holm_p"] < 0.05 and control["holm_p"] < 0.05,
        "at_least_14_native_sequence_wins": primary["win_tie_loss"][0] >= 14,
        "gains_without_seq8_positive": primary["mean_delta_without_seq8"] > 0 and control["mean_delta_without_seq8"] > 0,
        "boundary_f_not_below_native": cross["boundary_f"] >= native["boundary_f"],
        "pooled_false_alarms_not_above_native": cross["gt_empty_false_alarm_count"] <= native["gt_empty_false_alarm_count"],
        "undetected_returns_not_above_native": cross["undetected_returns"] <= native["undetected_returns"],
    }
    return {
        "primary_metric": "equal-sequence mean present-frame Dice",
        "cohort_summary": cohort,
        "contrasts": comparisons,
        "meaningful_improvement_screen": screen,
        "screen_passed": all(screen.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    summary = preflight(protocol)
    if args.preflight_only:
        print(json.dumps(summary, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --preflight-only is set")
    output = validate_output(args.output_dir)
    output_root = Path(protocol["output_root"]).resolve()
    if not output.is_relative_to(output_root) or output == output_root:
        raise RuntimeError("Choose a child folder under the frozen Drive output root")
    output_root.mkdir(parents=True, exist_ok=True)
    lock = (output_root / f".{output.name}.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("Another runner owns this attempt") from exc
    atexit.register(lock.close)
    if args.resume:
        setup = json.loads((output / "setup.json").read_text())
        if setup.get("durable_schema") != 1 or setup.get("protocol") != protocol or setup.get("protocol_sha256") != digest(args.protocol):
            raise RuntimeError("Resume setup differs from the frozen durable protocol")
        if (output / "completion.json").exists() or (output / "failure.json").exists():
            raise RuntimeError("Terminal attempts require review before resume")
    else:
        if output.exists():
            raise RuntimeError("Choose a new output directory or use --resume")
        output.mkdir(parents=True)
        write(output / "setup.json", {
            "durable_schema": 1,
            "protocol": protocol,
            "protocol_sha256": digest(args.protocol),
            "preflight": summary,
        })

    runtime = protocol["runtime"]
    torch.set_num_threads(runtime["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(runtime["seed"])
    videos = {**load_manifest(MANIFEST, "dev"), **load_manifest(MANIFEST, "test")}
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    native_data = json.loads(NATIVE_RESULTS.read_text())
    result_rows = {condition: {} for condition in CONDITIONS}
    summaries = {condition: {} for condition in CONDITIONS}
    runtimes = {condition: {} for condition in CONDITIONS}
    attention_audits = {}
    started = time.monotonic()
    try:
        for sequence_index, sequence in enumerate(SEQUENCES, start=1):
            frames = videos[sequence]
            prompt, native_rows, _ = native_reference(
                NATIVE_RESULTS.parent, native_data, native_protocol, sequence, frames
            )
            target_frames = list(range(prompt + 1, len(frames)))
            audits_by_condition = {}
            for condition in CONDITIONS:
                write(output / "progress.json", {
                    "status": "running",
                    "sequence_index": sequence_index,
                    "sequence": sequence,
                    "condition": condition,
                    "completed_conditions": sum(len(rows) for rows in result_rows.values()),
                    "total_conditions": len(SEQUENCES) * len(CONDITIONS),
                    "unique_post_prompt_targets": protocol["unique_post_prompt_targets"],
                    "condition_frame_outputs": protocol["unique_post_prompt_targets"] * len(CONDITIONS),
                })
                condition_dir = output / condition / sequence
                checkpoint = condition_dir / "condition_checkpoint.json"
                if args.resume and checkpoint.exists():
                    rows, raw_audit, receipt = load_condition_checkpoint(condition_dir, sequence, condition)
                else:
                    if args.resume:
                        archive_incomplete_condition(output, sequence, condition)
                    raw_audit = []
                    rows, receipt = run_condition(
                        "global", sequence, frames, prompt, native_rows,
                        CHECKPOINT_DEFAULT, output,
                        max_noncond_maps=6,
                        allocation="uniform",
                        condition_name=condition,
                        memory_reader_factory=lambda model, chosen=condition, sink=raw_audit: RecordingValueExchange(
                            model, mode=chosen, audit_sink=sink
                        ),
                        native_exact_required=(condition == "noop"),
                    )
                if [row["frame"] for row in rows] != target_frames:
                    raise RuntimeError(f"Frame coverage differs for {sequence}/{condition}")
                if any(row["present"] != native_rows[row["frame"]]["present"] for row in rows):
                    raise RuntimeError(f"Ground-truth presence differs for {sequence}/{condition}")
                if (receipt.get("targets") != len(target_frames)
                    or not receipt.get("parameter_hash_unchanged")
                    or not receipt.get("temporal_slots_restored")):
                    raise RuntimeError(f"Runtime invariants differ for {sequence}/{condition}")
                if condition == "noop":
                    if receipt.get("native_exact_frames_checked") != len(target_frames):
                        raise RuntimeError(f"Native parity count differs for {sequence}")
                    for row in rows:
                        reference = native_rows[row["frame"]]
                        for field in ("prediction_sha256", "stored_sha256", "scores", "absent_fp"):
                            if row[field] != reference[field]:
                                raise RuntimeError(f"Native/no-op mismatch: {sequence}/{row['frame']}/{field}")
                if not checkpoint.exists():
                    save_condition_checkpoint(condition_dir, sequence, condition, rows, raw_audit, receipt)
                result_rows[condition][sequence] = rows
                summaries[condition][sequence] = summarize(rows)
                runtimes[condition][sequence] = receipt
                audits_by_condition[condition] = raw_audit
                write(output / "partial_results.json", {
                    "status": "running",
                    "completed_conditions": sum(len(items) for items in result_rows.values()),
                    "summaries": summaries,
                    "runtimes": runtimes,
                    "frame_artifacts": "<condition>/<sequence>/frames.json",
                })
            attention_audits[sequence] = verify_fullprop_audits(audits_by_condition, target_frames)
            audit_dir = output / "sequence_audits" / sequence
            audit_file = audit_dir / "sequence_audit.json"
            if audit_file.exists():
                if json.loads(audit_file.read_text()) != {"sequence": sequence, "audit": attention_audits[sequence]}:
                    raise RuntimeError(f"Saved sequence audit differs: {sequence}")
            else:
                save_sequence_checkpoint(audit_dir, sequence, attention_audits[sequence])

        counts = count_fullprop_outputs({
            sequence: {condition: result_rows[condition][sequence] for condition in CONDITIONS}
            for sequence in SEQUENCES
        })
        if counts["unique_post_prompt_targets"] != protocol["unique_post_prompt_targets"]:
            raise RuntimeError("Final unique target count differs from the frozen cohort")
        if counts["condition_frame_outputs"] != protocol["unique_post_prompt_targets"] * len(CONDITIONS):
            raise RuntimeError("Final condition-frame count differs from the frozen cohort")
        comparison = cohort_comparison(summaries, protocol["statistics"])
        result = {
            "scope": "exploratory complete 21-sequence PolypGen cohort comparison; sequence-level independence unverified",
            "sequences": list(SEQUENCES),
            "conditions": list(CONDITIONS),
            "sequence_summaries": summaries,
            "sequence_rows": result_rows,
            "attention_audit": attention_audits,
            "comparison": comparison,
            "counts": counts,
            "runtime_seconds": time.monotonic() - started,
            "runtime_scope": "Current process invocation only; per-condition runtimes include resumed checkpoints.",
            "native_noop_exact": True,
        }
        for relative, expected in protocol["source_hashes"].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f"Source changed during cohort run: {relative}")
        write(output / "results.json", result)
        write(output / "runtime.json", runtimes)
        write(output / "completion.json", {
            "complete": True,
            "screen_passed": comparison["screen_passed"],
            "unique_post_prompt_targets": counts["unique_post_prompt_targets"],
            "condition_frame_outputs": counts["condition_frame_outputs"],
            "sequence_targets": counts["sequence_targets"],
            "results_sha256": digest(output / "results.json"),
        })
        write(output / "progress.json", {"status": "complete", **counts})
    except BaseException as exc:
        write(output / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        write(output / "progress.json", {
            "status": "failed",
            "error": str(exc),
            "completed_conditions": sum(len(rows) for rows in result_rows.values()),
        })
        raise


if __name__ == "__main__":
    main()
