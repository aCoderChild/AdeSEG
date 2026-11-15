"""Four-arm ten-map token-allocation study on the complete PolypGen cohort."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from scripts.memory_token_io import digest, validate_output
from scripts.run_region_constrained_tome_gate import native_reference
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import summarize
from scripts.run_ten_map_allocation_tome_gate import ARMS, MAX_NONCOND_MAPS, PROTOCOL
from scripts.run_wider_history_tome_gate import (
    CHECKPOINT_DEFAULT,
    NATIVE_PROTOCOL,
    NATIVE_RESULTS,
    NATIVE_SETUP,
    SAM2_REFERENCE,
    SOURCES as GATE_SOURCES,
    _assert_read_integrity,
    run_condition,
    verify_native_setup,
)


EXCLUDED = {"seq1", "seq7"}
FULL_SEQUENCES = tuple(json.loads(PROTOCOL.read_text())["scope"]["sequences"])
CONTRASTS = (
    ("recent_dense_regional_minus_native", "recent_dense_regional", "native"),
    ("recent_dense_regional_minus_recent_dense_generic", "recent_dense_regional", "recent_dense_generic"),
    ("recent_dense_generic_minus_uniform_generic", "recent_dense_generic", "uniform_generic"),
    ("recent_dense_regional_minus_uniform_regional", "recent_dense_regional", "uniform_regional"),
    ("uniform_regional_minus_uniform_generic", "uniform_regional", "uniform_generic"),
)


def compare_arms(sequence_summaries):
    names = sorted(sequence_summaries["native"])
    if len(names) != 21 or set(names) & EXCLUDED:
        raise RuntimeError("Expected the 21 eligible sequences only")
    contrast_results = {}
    for contrast_name, left_arm, right_arm in CONTRASTS:
        left = {name: sequence_summaries[left_arm][name]["present_dice"] for name in names}
        right = {name: sequence_summaries[right_arm][name]["present_dice"] for name in names}
        deltas = {name: left[name] - right[name] for name in names}
        result = paired_wilcoxon(left, right)
        result.update({
            "mean_sequence_delta": float(np.mean(list(deltas.values()))),
            "ci95_sequence_bootstrap": sequence_bootstrap_interval(
                deltas, seed=20261112
            ),
            "sequence_win_tie_loss": [
                sum(delta > 1e-12 for delta in deltas.values()),
                sum(abs(delta) <= 1e-12 for delta in deltas.values()),
                sum(delta < -1e-12 for delta in deltas.values()),
            ],
            "deltas_by_sequence": deltas,
        })
        contrast_results[contrast_name] = result
    adjusted = holm_adjust({key: row["p"] for key, row in contrast_results.items()})
    for key, row in contrast_results.items():
        row["holm_p_five_planned_contrasts"] = adjusted[key]

    cohort = {}
    for arm, sequences in sequence_summaries.items():
        present_dice = [row["present_dice"] for row in sequences.values()]
        present_iou = [row["iou"] for row in sequences.values()]
        boundary_f = [row["boundary_f"] for row in sequences.values()]
        empty_count = sum(row["gt_empty_false_alarm_count"] for row in sequences.values())
        empty_targets = sum(row["gt_empty_targets"] for row in sequences.values())
        return_count = sum(row["return_count"] for row in sequences.values())
        undetected = sum(row["undetected_returns"] for row in sequences.values())
        cohort[arm] = {
            "equal_sequence_present_dice": float(np.mean(present_dice)),
            "equal_sequence_iou": float(np.mean(present_iou)),
            "equal_sequence_boundary_f": float(np.mean(boundary_f)),
            "gt_empty_false_alarms": empty_count,
            "gt_empty_targets": empty_targets,
            "gt_empty_false_alarm_rate": empty_count / empty_targets if empty_targets else None,
            "returns": return_count,
            "undetected_returns": undetected,
            "return_coverage": (return_count - undetected) / return_count if return_count else None,
        }

    primary = contrast_results["recent_dense_regional_minus_native"]
    regional_control = contrast_results["recent_dense_regional_minus_recent_dense_generic"]
    without_seq8 = [name for name in names if name != "seq8"]
    gain_without_seq8 = float(np.mean([
        sequence_summaries["recent_dense_regional"][name]["present_dice"]
        - sequence_summaries["native"][name]["present_dice"]
        for name in without_seq8
    ]))
    screen = {
        "mean_dice_at_least_0_74": cohort["recent_dense_regional"]["equal_sequence_present_dice"] >= 0.74,
        "gain_vs_native_at_least_0_01": primary["mean_sequence_delta"] >= 0.01,
        "at_least_14_sequence_wins_vs_native": primary["sequence_win_tie_loss"][0] >= 14,
        "primary_ci_lower_positive": primary["ci95_sequence_bootstrap"][0] > 0,
        "primary_holm_p_below_0_05": primary["holm_p_five_planned_contrasts"] < 0.05,
        "region_method_beats_recent_dense_generic": regional_control["mean_sequence_delta"] > 0,
        "region_control_ci_lower_positive": regional_control["ci95_sequence_bootstrap"][0] > 0,
        "region_control_holm_p_below_0_05": regional_control["holm_p_five_planned_contrasts"] < 0.05,
        "no_increase_in_pooled_gt_empty_false_alarms": (
            cohort["recent_dense_regional"]["gt_empty_false_alarms"]
            <= cohort["native"]["gt_empty_false_alarms"]
        ),
        "gain_without_seq8_positive": gain_without_seq8 > 0,
        "gain_without_seq8": gain_without_seq8,
    }
    screen["passed"] = all(
        value for key, value in screen.items() if key != "gain_without_seq8"
    )
    return {
        "cohort_summary": cohort,
        "planned_contrasts": contrast_results,
        "screen": screen,
        "statistical_unit": "sequence; equal-sequence mean of post-prompt GT-present frame metrics",
        "interpretation": "Exploratory only: PolypGen informed candidate design; no confirmatory inference is claimed.",
    }


def _assert_all_arm_alignment(rows_by_arm, sequence):
    reference = rows_by_arm["uniform_generic"]
    for arm, rows in rows_by_arm.items():
        if [row["frame"] for row in rows] != [row["frame"] for row in reference]:
            raise RuntimeError(f"Post-prompt frame coverage differs for {sequence}/{arm}")
    for target_idx, ref_row in enumerate(reference):
        for arm, rows in rows_by_arm.items():
            arm_row = rows[target_idx]
            if arm_row["present"] != ref_row["present"]:
                raise RuntimeError(f"GT-presence differs for {sequence}/{arm}")
            for ref_read, arm_read in zip(
                ref_row["attention_reads"], arm_row["attention_reads"], strict=True
            ):
                ref_ids = [(x["frame"], x["t_pos"]) for x in ref_read["frames"]]
                arm_ids = [(x["frame"], x["t_pos"]) for x in arm_read["frames"]]
                if ref_ids != arm_ids:
                    raise RuntimeError(f"Attention reads use different source frames: {sequence}/{arm}")


def _load_resumable_arm(source_dir, arm, sequence, expected_frames, allocation):
    """Reuse only a complete arm/sequence whose ledger agrees with its rows."""
    source_dir = Path(source_dir)
    summaries_path = source_dir / "sequence_summaries.json"
    runtimes_path = source_dir / "runtime.json"
    frames_path = source_dir / arm / sequence / "frames.json"
    if not summaries_path.is_file():
        return None
    summaries = json.loads(summaries_path.read_text()).get(arm, {})
    if sequence not in summaries:
        return None
    if not frames_path.is_file() or not runtimes_path.is_file():
        raise RuntimeError(f"Resume ledger is incomplete for {sequence}/{arm}")

    rows = json.loads(frames_path.read_text())
    if [row.get("frame") for row in rows] != list(expected_frames):
        raise RuntimeError(f"Saved resume rows are incomplete or misordered for {sequence}/{arm}")
    for row in rows:
        if row.get("condition_name") != arm or row.get("allocation") != allocation:
            raise RuntimeError(f"Saved resume rows have wrong arm metadata for {sequence}/{arm}")
        reads = row.get("attention_reads", [])
        if [read.get("layer") for read in reads] != [0, 1, 2, 3]:
            raise RuntimeError(f"Saved resume row lacks four audited layers for {sequence}/{arm}")
        for read in reads:
            _assert_read_integrity(read, expected_max_noncond=MAX_NONCOND_MAPS)

    if summarize(rows) != summaries[sequence]:
        raise RuntimeError(f"Saved resume summary does not match rows for {sequence}/{arm}")
    runtimes = json.loads(runtimes_path.read_text()).get(arm, {})
    if sequence not in runtimes or not isinstance(runtimes[sequence], dict):
        raise RuntimeError(f"Saved resume runtime is missing for {sequence}/{arm}")
    runtime = runtimes[sequence]
    if (
        runtime.get("sequence") != sequence
        or runtime.get("targets") != len(rows)
        or runtime.get("attention_layer_calls") != 4 * len(rows)
        or runtime.get("parameter_hash_unchanged") is not True
        or runtime.get("temporal_slots_restored") is not True
    ):
        raise RuntimeError(f"Saved resume runtime receipt is invalid for {sequence}/{arm}")
    return rows, runtime


def _validate_resume_source(source_dir, expected_setup, current_sources):
    """Reject resumption if the frozen protocol or method inputs differ."""
    source_dir = Path(source_dir).resolve()
    setup_path = source_dir / "setup.json"
    if not setup_path.is_file():
        raise RuntimeError("Resume source has no setup receipt")
    old = json.loads(setup_path.read_text())
    for key, value in expected_setup.items():
        if old.get(key) != value:
            raise RuntimeError(f"Resume source differs from current setup field: {key}")
    old_sources = old.get("source_hashes", {})
    runner_path = "scripts/run_ten_map_allocation_tome_full.py"
    for path, checksum in current_sources.items():
        if path != runner_path and old_sources.get(path) != checksum:
            raise RuntimeError(f"Resume source-code hash differs: {path}")
    if runner_path not in old_sources:
        raise RuntimeError("Resume source does not record the original full-runner hash")
    return {
        "source_dir": str(source_dir),
        "source_setup_sha256": digest(setup_path),
        "original_full_runner_sha256": old_sources[runner_path],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-from", type=Path,
        help="Reuse complete sequence/arm receipts from an interrupted compatible run",
    )
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    protocol = json.loads(PROTOCOL.read_text())
    manifest = ROOT / "data/polypgen_sequence.jsonl"
    sequences = tuple(protocol["scope"]["sequences"])
    if sequences != FULL_SEQUENCES or len(set(sequences)) != 21 or set(sequences) & EXCLUDED:
        raise RuntimeError("Protocol sequence set differs from the full eligible cohort")
    if digest(args.checkpoint) != "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69":
        raise RuntimeError("Checkpoint digest differs from the verified native checkpoint")
    native_setup = verify_native_setup(args.checkpoint, manifest)
    if digest(manifest) != native_setup["manifest_sha256"]:
        raise RuntimeError("Manifest differs from the archived native reference")
    from scripts.run_wider_history_tome_gate import activate_reference_repo

    activate_reference_repo()

    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    native_results = json.loads(NATIVE_RESULTS.read_text())
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    native_refs, prompts, native_receipts = {}, {}, {}
    for sequence in sequences:
        prompt, rows, receipt = native_reference(
            NATIVE_RESULTS.parent, native_results, native_protocol, sequence, videos[sequence]
        )
        native_refs[sequence], prompts[sequence], native_receipts[sequence] = rows, prompt, receipt

    sources = {
        **{source: digest(ROOT / source) for source in GATE_SOURCES},
        "scripts/run_ten_map_allocation_tome_gate.py": digest(ROOT / "scripts/run_ten_map_allocation_tome_gate.py"),
        "scripts/run_ten_map_allocation_tome_full.py": digest(Path(__file__)),
        "evaluation/statistics.py": digest(ROOT / "evaluation/statistics.py"),
        "scripts/false_alarm_statistics.py": digest(ROOT / "scripts/false_alarm_statistics.py"),
    }
    arm_names = [name for name, _, _ in ARMS]
    setup_identity = {
        "protocol_sha256": digest(PROTOCOL),
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(manifest),
        "native_setup_sha256": digest(NATIVE_SETUP),
        "native_results_sha256": digest(NATIVE_RESULTS),
        "sequences": list(sequences),
        "excluded": sorted(EXCLUDED),
        "arms": arm_names,
        "maximum_nonconditioning_maps": MAX_NONCOND_MAPS,
        "device": "cpu",
    }
    resume_info = None
    if args.resume_from is not None:
        resume_info = _validate_resume_source(
            args.resume_from, setup_identity, sources
        )
    write(args.output_dir / "setup.json", {
        **setup_identity,
        "source_hashes": sources,
        "native_reference_receipts": native_receipts,
        "resumed_from": resume_info,
        "status": "exploratory full-video run in progress",
    })

    sequence_summaries = {"native": {}, **{arm: {} for arm in arm_names}}
    runtime = {arm: {} for arm in arm_names}
    frame_rows = {arm: {} for arm in arm_names}
    completed = []
    if resume_info is not None:
        for sequence in sequences:
            expected_frames = range(prompts[sequence] + 1, len(videos[sequence]))
            for arm_name, matching, allocation in ARMS:
                saved = _load_resumable_arm(
                    args.resume_from, arm_name, sequence,
                    expected_frames, allocation,
                )
                if saved is None:
                    continue
                rows, elapsed = saved
                source_native = native_refs[sequence]
                if any(row["present"] != source_native[row["frame"]]["present"] for row in rows):
                    raise RuntimeError(f"Resume GT-presence labels differ for {sequence}/{arm_name}")
                destination = args.output_dir / arm_name / sequence
                destination.mkdir(parents=True, exist_ok=False)
                shutil.copy2(
                    Path(args.resume_from) / arm_name / sequence / "frames.json",
                    destination / "frames.json",
                )
                frame_rows[arm_name][sequence] = rows
                sequence_summaries[arm_name][sequence] = summarize(rows)
                runtime[arm_name][sequence] = elapsed
                completed.append(f"{sequence}/{arm_name}")
        write(args.output_dir / "sequence_summaries.json", sequence_summaries)
        write(args.output_dir / "runtime.json", runtime)
    write(args.output_dir / "progress.json", {
        "status": "running", "sequence_index": 0, "sequence_total": len(sequences),
        "arm": None, "completed_sequence_arms": completed,
        "resumed_sequence_arms": len(completed),
    })

    for sequence_index, sequence in enumerate(sequences, start=1):
        native_rows = [native_refs[sequence][frame] for frame in sorted(native_refs[sequence])]
        sequence_summaries["native"][sequence] = summarize(native_rows)
        current_rows = {
            arm: frame_rows[arm][sequence]
            for arm in arm_names if sequence in frame_rows[arm]
        }
        for arm_name, matching, allocation in ARMS:
            if sequence in frame_rows[arm_name]:
                rows = frame_rows[arm_name][sequence]
                elapsed = runtime[arm_name][sequence]
            else:
                rows, elapsed = run_condition(
                    matching,
                    sequence,
                    videos[sequence],
                    prompts[sequence],
                    native_refs[sequence],
                    args.checkpoint,
                    args.output_dir,
                    max_noncond_maps=MAX_NONCOND_MAPS,
                    allocation=allocation,
                    condition_name=arm_name,
                )
                frame_rows[arm_name][sequence] = rows
                sequence_summaries[arm_name][sequence] = summarize(rows)
                runtime[arm_name][sequence] = elapsed
                completed.append(f"{sequence}/{arm_name}")
            frame_rows[arm_name][sequence] = rows
            sequence_summaries[arm_name][sequence] = summarize(rows)
            runtime[arm_name][sequence] = elapsed
            current_rows[arm_name] = rows
            write(args.output_dir / "sequence_summaries.json", sequence_summaries)
            write(args.output_dir / "runtime.json", runtime)
            write(args.output_dir / "progress.json", {
                "status": "running",
                "sequence_index": sequence_index,
                "sequence_total": len(sequences),
                "sequence": sequence,
                "arm": arm_name,
                "targets_completed": len(rows),
                "targets_total": len(videos[sequence]) - prompts[sequence] - 1,
                "completed_sequence_arms": completed,
            })
        _assert_all_arm_alignment(current_rows, sequence)

    comparison = compare_arms(sequence_summaries)
    write(args.output_dir / "results.json", {
        "sequence_summaries": sequence_summaries,
        "runtime_seconds_by_sequence": runtime,
        "comparison": comparison,
        "frame_rows": frame_rows,
    })
    cohort = comparison["cohort_summary"]
    lines = [
        "# Ten-map ToMe allocation comparison — PolypGen (exploratory)",
        "",
        "Complete post-prompt propagation; 21 eligible sequences; seq1 and seq7 excluded.",
        "Native values are reused from the hash-verified archived native run. All four treatment arms read identical source frames per target.",
        "The study is exploratory because PolypGen informed candidate design; it is not untouched confirmation.",
        "",
        "| Condition | Present Dice | IoU | Boundary F | GT-empty false alarms | Returns detected |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("native", *arm_names):
        row = cohort[arm]
        lines.append(
            f"| {arm} | {row['equal_sequence_present_dice']:.6f} | "
            f"{row['equal_sequence_iou']:.6f} | {row['equal_sequence_boundary_f']:.6f} | "
            f"{row['gt_empty_false_alarms']}/{row['gt_empty_targets']} | "
            f"{row['returns']-row['undetected_returns']}/{row['returns']} |"
        )
    lines += ["", "## Paired sequence contrasts", "", "| Contrast | Mean Δ Dice | 95% sequence-bootstrap CI | p | Holm p (5 contrasts) | Wins/ties/losses |", "|---|---:|---:|---:|---:|---:|"]
    for key, row in comparison["planned_contrasts"].items():
        ci = row["ci95_sequence_bootstrap"]
        wtl = "/".join(str(x) for x in row["sequence_win_tie_loss"])
        lines.append(
            f"| {key.replace('_', ' ')} | {row['mean_sequence_delta']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {row['p']:.6g} | "
            f"{row['holm_p_five_planned_contrasts']:.6g} | {wtl} |"
        )
    lines += [
        "",
        f"Frozen resource screen: **{'PASS' if comparison['screen']['passed'] else 'FAIL'}**.",
        f"Primary recent-dense regional Dice gain without seq8: {comparison['screen']['gain_without_seq8']:+.6f}.",
        "All intervals and tests are descriptive after candidate selection on this cohort; no confirmatory claim follows from a pass.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    write(args.output_dir / "completion.json", {
        "complete": True,
        "screen_passed": comparison["screen"]["passed"],
        "sequences": list(sequences),
        "arms": arm_names,
        "status": "exploratory full-video study complete; untouched-data confirmation still required",
    })
    write(args.output_dir / "progress.json", {
        "status": "complete", "sequence_index": len(sequences),
        "sequence_total": len(sequences), "screen_passed": comparison["screen"]["passed"],
    })
    print(f"Completed ten-map allocation comparison: {args.output_dir}")


if __name__ == "__main__":
    main()
