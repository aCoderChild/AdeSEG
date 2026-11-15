"""Full 21-sequence fixed-budget wider-history ToMe evaluation on Google Drive."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, validate_output
from scripts.run_region_constrained_tome_gate import native_reference
from scripts.run_regional_refinement import write
from scripts.run_wider_history_tome_gate import (
    CHECKPOINT_DEFAULT,
    NATIVE_PROTOCOL,
    NATIVE_RESULTS,
    NATIVE_SETUP,
    PROTOCOL,
    SAM2_REFERENCE,
    SOURCES as GATE_SOURCES,
    run_condition,
    verify_native_setup,
)


CONDITIONS = ("global", "regional")
EXCLUDED = {"seq1", "seq7"}


def sequence_summary(rows):
    return propagation_summary(rows)


def compare_sequences(candidate, generic, native):
    names = sorted(native)
    if set(candidate) != set(names) or set(generic) != set(names):
        raise RuntimeError("All three conditions must cover the same 21 sequences")
    deltas_native, deltas_generic = {}, {}
    sequence_rows = {}
    wins = ties = losses = 0
    for name in names:
        regional = candidate[name]["present_dice"]
        global_score = generic[name]["present_dice"]
        native_score = native[name]["present_dice"]
        delta = regional - native_score
        deltas_native[name] = delta
        deltas_generic[name] = regional - global_score
        wins += delta > 1e-12
        ties += abs(delta) <= 1e-12
        losses += delta < -1e-12
        sequence_rows[name] = {
            "native_present_dice": native_score,
            "generic_12frame_present_dice": global_score,
            "regional_12frame_present_dice": regional,
            "regional_minus_native": delta,
            "regional_minus_generic": regional - global_score,
            "native_gt_empty_false_alarms": native[name]["gt_empty_false_alarm_count"],
            "generic_gt_empty_false_alarms": generic[name]["gt_empty_false_alarm_count"],
            "regional_gt_empty_false_alarms": candidate[name]["gt_empty_false_alarm_count"],
            "native_gt_empty_targets": native[name]["gt_empty_targets"],
            "generic_gt_empty_targets": generic[name]["gt_empty_targets"],
            "regional_gt_empty_targets": candidate[name]["gt_empty_targets"],
            "native_return_count": native[name]["return_count"],
            "generic_return_count": generic[name]["return_count"],
            "regional_return_count": candidate[name]["return_count"],
            "native_undetected_returns": native[name]["undetected_returns"],
            "generic_undetected_returns": generic[name]["undetected_returns"],
            "regional_undetected_returns": candidate[name]["undetected_returns"],
        }

    contrast_native = paired_wilcoxon(
        {name: candidate[name]["present_dice"] for name in names},
        {name: native[name]["present_dice"] for name in names},
    )
    contrast_generic = paired_wilcoxon(
        {name: candidate[name]["present_dice"] for name in names},
        {name: generic[name]["present_dice"] for name in names},
    )
    p_adjusted = holm_adjust({
        "regional_minus_native": contrast_native["p"],
        "regional_minus_generic": contrast_generic["p"],
    })
    for key, contrast, deltas in (
        ("regional_minus_native", contrast_native, deltas_native),
        ("regional_minus_generic", contrast_generic, deltas_generic),
    ):
        contrast.update({
            "mean_sequence_delta": sum(deltas.values()) / len(deltas),
            "ci95_sequence_bootstrap": sequence_bootstrap_interval(
                deltas, seed=20261112
            ),
            "holm_p_two_contrasts": p_adjusted[key],
        })

    totals = {}
    for condition, rows_by_sequence in (
        ("native", native),
        ("generic", generic),
        ("regional", candidate),
    ):
        false_alarms = sum(row["gt_empty_false_alarm_count"] for row in rows_by_sequence.values())
        empty_targets = sum(row["gt_empty_targets"] for row in rows_by_sequence.values())
        present_means = [row["present_dice"] for row in rows_by_sequence.values()]
        totals[condition] = {
            "equal_sequence_mean_present_dice": sum(present_means) / len(present_means),
            "gt_empty_false_alarms": false_alarms,
            "gt_empty_targets": empty_targets,
            "gt_empty_false_alarm_rate": false_alarms / empty_targets if empty_targets else None,
            "sequence_win_tie_loss_vs_native": (
                [wins, ties, losses] if condition == "regional" else None
            ),
        }

    without_seq8 = [name for name in names if name != "seq8"]
    without_seq8_delta = sum(deltas_native[name] for name in without_seq8) / len(without_seq8)
    screen = {
        "regional_gain_native_at_least_0_01": totals["regional"]["equal_sequence_mean_present_dice"]
        - totals["native"]["equal_sequence_mean_present_dice"] >= 0.01,
        "regional_mean_above_generic": totals["regional"]["equal_sequence_mean_present_dice"]
        > totals["generic"]["equal_sequence_mean_present_dice"],
        "at_least_14_sequence_wins_native": wins >= 14,
        "both_holm_p_below_0_05": p_adjusted["regional_minus_native"] < 0.05
        and p_adjusted["regional_minus_generic"] < 0.05,
        "both_bootstrap_lower_bounds_positive": contrast_native["ci95_sequence_bootstrap"][0] > 0
        and contrast_generic["ci95_sequence_bootstrap"][0] > 0,
        "no_increase_pooled_empty_frame_false_alarms": totals["regional"]["gt_empty_false_alarms"]
        <= totals["native"]["gt_empty_false_alarms"],
        "gain_without_seq8": without_seq8_delta,
    }
    screen["passed"] = all(
        value for key, value in screen.items()
        if key != "gain_without_seq8"
    )
    return {
        "sequence_metrics": sequence_rows,
        "cohort_summary": totals,
        "planned_contrasts": {
            "regional_minus_native": contrast_native,
            "regional_minus_generic": contrast_generic,
        },
        "predeclared_meaningful_screen": screen,
        "statistical_unit": "sequence; paired present-frame Dice mean per sequence",
        "warning": "PolypGen sequence-level patient independence is not verified; this is exploratory, not independent confirmation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    manifest = ROOT / "data/polypgen_sequence.jsonl"
    protocol = json.loads(PROTOCOL.read_text())
    sequences = tuple(protocol["scope"]["sequences"])
    if len(sequences) != 21 or set(sequences) & EXCLUDED or len(set(sequences)) != 21:
        raise RuntimeError("Protocol must include exactly the 21 eligible sequences")
    if digest(args.checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint digest differs from the frozen protocol")
    native_setup = verify_native_setup(args.checkpoint, manifest)
    if digest(manifest) != native_setup["manifest_sha256"]:
        raise RuntimeError("Manifest differs from the archived native reference")
    for source, expected in native_setup.get("sources", {}).items():
        if source.startswith("external/sam2/"):
            relative = source.removeprefix("external/sam2/")
            if digest(SAM2_REFERENCE / relative) != expected:
                raise RuntimeError(f"SAM2 source differs from archived reference: {relative}")

    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    native_results = json.loads(NATIVE_RESULTS.read_text())
    native_protocol = json.loads(NATIVE_PROTOCOL.read_text())
    native_refs, prompts, receipts = {}, {}, {}
    for name in sequences:
        if name not in videos:
            raise RuntimeError(f"Sequence missing from PolypGen manifest: {name}")
        prompt, rows, receipt = native_reference(
            NATIVE_RESULTS.parent, native_results, native_protocol, name, videos[name]
        )
        native_refs[name], prompts[name], receipts[name] = rows, prompt, receipt

    source_hashes = {
        **{source: digest(ROOT / source) for source in GATE_SOURCES},
        "scripts/run_wider_history_tome_full.py": digest(
            ROOT / "scripts/run_wider_history_tome_full.py"
        ),
        "evaluation/statistics.py": digest(ROOT / "evaluation/statistics.py"),
        "scripts/false_alarm_statistics.py": digest(ROOT / "scripts/false_alarm_statistics.py"),
    }
    write(args.output_dir / "setup.json", {
        "protocol_sha256": digest(PROTOCOL),
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(manifest),
        "native_setup_sha256": digest(NATIVE_SETUP),
        "native_results_sha256": digest(NATIVE_RESULTS),
        "sequences": list(sequences),
        "excluded": {"seq1": "no positive first-frame prompt", "seq7": "no positive first-frame prompt"},
        "conditions": list(CONDITIONS),
        "device": "cpu",
        "native_reference_receipts": receipts,
        "source_hashes": source_hashes,
        "status": "full-video efficacy run in progress",
    })
    write(args.output_dir / "progress.json", {
        "status": "running", "sequence_index": 0, "sequence_total": len(sequences),
        "condition": None, "completed_sequences": [],
    })

    rows_by_condition = {condition: {} for condition in CONDITIONS}
    sequence_summaries = {condition: {} for condition in ("native", *CONDITIONS)}
    runtime_summaries = {condition: {} for condition in CONDITIONS}
    for name in sequences:
        native_rows = [native_refs[name][frame] for frame in sorted(native_refs[name])]
        sequence_summaries["native"][name] = sequence_summary(native_rows)
        for condition in CONDITIONS:
            rows, runtime = run_condition(
                condition, name, videos[name], prompts[name], native_refs[name],
                args.checkpoint, args.output_dir,
            )
            rows_by_condition[condition][name] = rows
            sequence_summaries[condition][name] = sequence_summary(rows)
            runtime_summaries[condition][name] = runtime
            write(args.output_dir / "sequence_summaries.json", sequence_summaries)
            write(args.output_dir / "runtime_summaries.json", runtime_summaries)
            write(args.output_dir / "progress.json", {
                "status": "running",
                "sequence_index": sequences.index(name) + 1,
                "sequence_total": len(sequences),
                "sequence": name,
                "condition": condition,
                "targets_completed": len(rows),
                "targets_total": len(videos[name]) - prompts[name] - 1,
                "completed_conditions_for_sequence": [
                    arm for arm in CONDITIONS if name in rows_by_condition[arm]
                ],
            })

    for name in sequences:
        generic = {row["frame"]: row for row in rows_by_condition["global"][name]}
        regional = {row["frame"]: row for row in rows_by_condition["regional"][name]}
        if generic.keys() != regional.keys() or generic.keys() != native_refs[name].keys():
            raise RuntimeError(f"Full-video target coverage differs for {name}")
        for frame in generic:
            for g_read, r_read in zip(
                generic[frame]["attention_reads"], regional[frame]["attention_reads"], strict=True
            ):
                if g_read["frames"] != r_read["frames"] or g_read["pointer_keys"] != r_read["pointer_keys"]:
                    raise RuntimeError(f"Matched budget/pointer mismatch at {name}/{frame}")

    native_summaries = sequence_summaries["native"]
    comparison = compare_sequences(
        sequence_summaries["regional"], sequence_summaries["global"], native_summaries
    )
    write(args.output_dir / "results.json", {
        "sequence_summaries": sequence_summaries,
        "runtime_summaries": runtime_summaries,
        "comparison": comparison,
        "frame_rows": rows_by_condition,
    })
    lines = [
        "# Fixed-budget wider-history ToMe — PolypGen full evaluation",
        "",
        "All 21 eligible sequences; seq1 and seq7 excluded. One first-frame prompt; complete post-prompt propagation.",
        "Native SAM2 values are reused from the hash-verified archived native run; no native rerun was performed.",
        "",
        "| Condition | Equal-sequence present Dice | GT-empty false alarms | GT-empty frames | False-alarm rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in ("native", "global", "regional"):
        row = comparison["cohort_summary"][condition]
        lines.append(
            f"| {condition} | {row['equal_sequence_mean_present_dice']:.6f} | "
            f"{row['gt_empty_false_alarms']} | {row['gt_empty_targets']} | "
            f"{row['gt_empty_false_alarm_rate']:.4f} |"
        )
    lines += [
        "",
        "| Sequence | Native | Generic 12-frame | Regional 12-frame | Regional − native |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in comparison["sequence_metrics"].items():
        lines.append(
            f"| {name} | {row['native_present_dice']:.6f} | "
            f"{row['generic_12frame_present_dice']:.6f} | "
            f"{row['regional_12frame_present_dice']:.6f} | "
            f"{row['regional_minus_native']:+.6f} |"
        )
    lines += [
        "",
        "## Paired sequence comparisons",
        "",
        "| Contrast | Mean Dice difference | 95% sequence-bootstrap CI | Wilcoxon p | Holm p |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in comparison["planned_contrasts"].items():
        ci = row["ci95_sequence_bootstrap"]
        lines.append(
            f"| {name.replace('_', ' ')} | {row['mean_sequence_delta']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {row['p']:.6g} | "
            f"{row['holm_p_two_contrasts']:.6g} |"
        )
    screen = comparison["predeclared_meaningful_screen"]
    lines += [
        "",
        f"Predeclared screen: **{'PASS' if screen['passed'] else 'FAIL'}**.",
        f"Sequence wins/ties/losses vs native: {comparison['cohort_summary']['regional']['sequence_win_tie_loss_vs_native']}",
        f"Gain without seq8: {screen['gain_without_seq8']:+.6f} Dice.",
        "",
        "Interpret PolypGen sequence-level statistics as exploratory because patient independence by sequence is unverified.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    write(args.output_dir / "completion.json", {
        "complete": True,
        "sequences": len(sequences),
        "excluded": sorted(EXCLUDED),
        "screen_passed": screen["passed"],
        "status": "completed full-video evaluation; external confirmation still required",
    })
    write(args.output_dir / "progress.json", {
        "status": "complete", "sequence_index": len(sequences),
        "sequence_total": len(sequences), "screen_passed": screen["passed"],
    })
    print(f"Completed 21-sequence evaluation: {args.output_dir}")


if __name__ == "__main__":
    main()
