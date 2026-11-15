"""Audit a completed cross-frame ToMe cohort against saved frame-wise YOLO masks.

This is an offline saved-output comparison; it runs no SAM2 or YOLO inference.
It verifies matching frames, current GT and YOLO-mask hashes, then reports
equal-sequence present-frame Dice and descriptive paired sequence uncertainty.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from evaluation.statistics import (
    holm_adjust,
    paired_wilcoxon,
    sequence_bootstrap_interval,
)
from scripts.audit_matched_framewise_comparison import (
    load_manifest,
    mean,
    read_json,
    sha256,
)
from scripts.false_alarm_statistics import propagation_summary


CONDITIONS = ("noop", "local", "cross_frame")
LABELS = {
    "noop": "Native SAM2",
    "local": "Within-frame ToMe control",
    "cross_frame": "Cross-frame ToMe",
    "framewise_yolo": "Saved frame-wise YOLO-box masks",
}


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = [row for row in rows if row["present"]]
    empty = [row for row in rows if not row["present"]]
    temporal = propagation_summary(rows)
    return {
        "targets": len(rows),
        "present_targets": len(present),
        "empty_targets": len(empty),
        "present_dice": mean([float(row["scores"]["dice"]) for row in present]),
        "present_boundary_f": mean(
            [float(row["scores"]["boundary_f"]) for row in present]
        ),
        "false_alarms": sum(bool(row["absent_fp"]) for row in empty),
        "return_count": int(temporal["return_count"]),
        "undetected_returns": int(temporal["undetected_returns"]),
        "return_events": temporal["returns"],
    }


def _verify_frozen_inputs(args: argparse.Namespace, setup: dict[str, Any]) -> dict[str, Any]:
    protocol = read_json(args.protocol)
    if sha256(args.protocol) != setup["protocol_sha256"]:
        raise ValueError("Run setup protocol hash differs from the supplied frozen protocol")
    if protocol != setup["protocol"]:
        raise ValueError("Embedded protocol differs from the supplied frozen protocol")
    if sha256(args.manifest) != protocol["manifest_sha256"]:
        raise ValueError("Current PolypGen manifest hash differs from the frozen protocol")
    source_mismatches = [
        name for name, expected in protocol["source_hashes"].items()
        if not (ROOT / name).is_file() or sha256(ROOT / name) != expected
    ]
    if source_mismatches:
        raise ValueError(f"Frozen source hash mismatch: {source_mismatches[:3]}")
    return protocol


def _candidate_settings(protocol: dict[str, Any]) -> dict[str, Any]:
    native_setup = read_json(Path(protocol["native_reference"]["setup"]))
    config_path = ROOT / "external/sam2/sam2" / native_setup["model_config"]
    config_text = config_path.read_text()
    image_size_match = re.search(r"(?m)^\s*image_size:\s*(\d+)\s*$", config_text)
    if image_size_match is None:
        raise ValueError(f"Could not verify SAM2 image_size in {config_path}")
    return {
        "sam2_checkpoint_sha256": protocol["checkpoint_sha256"],
        "sam2_reference_revision": protocol["sam2_reference_revision"],
        "sam2_model_config": native_setup["model_config"],
        "sam2_model_config_sha256": sha256(config_path),
        "sam2_image_size": int(image_size_match.group(1)),
        "prompt_policy": "one initial native box prompt per sequence; no later detector boxes",
        "prompt_frame_indices": {
            seq: int(row["prompt"]) for seq, row in protocol["sequence_targets"].items()
        },
        "runtime": protocol["runtime"],
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.candidate_run_dir
    setup = read_json(run_dir / "setup.json")
    protocol = _verify_frozen_inputs(args, setup)
    baseline_audit = args.baseline_audit_dir
    baseline_scores = read_json(baseline_audit / "saved_mask_recomputation.json")
    baseline_reaggregation = read_json(
        baseline_audit / "recorded_baseline_reaggregation.json"
    )
    if (
        protocol["source_hashes"]["evaluation/segmentation.py"]
        != baseline_scores["evaluator_sources"]["evaluation/segmentation.py"]
    ):
        raise ValueError("Candidate and YOLO saved masks use different evaluator revisions")

    completion_path = run_dir / "completion.json"
    results_path = run_dir / "results.json"
    if completion_path.exists() and results_path.exists():
        completion = read_json(completion_path)
        results = read_json(results_path)
        if not completion.get("complete") or sha256(results_path) != completion["results_sha256"]:
            raise ValueError("Full-cohort results do not match their completion receipt")
        if results.get("native_noop_exact") is not True:
            raise ValueError("Cohort results do not certify exact native/no-op parity")
        summaries = results["sequence_summaries"]
        sequences = list(results["sequences"])
        full = True
    elif args.allow_partial:
        results = read_json(run_dir / "partial_results.json")
        summaries = results["summaries"]
        common = set.intersection(*(set(summaries.get(c, {})) for c in CONDITIONS))
        sequences = [seq for seq in protocol["sequences"] if seq in common]
        if not sequences:
            raise ValueError("No complete three-condition sequence triplets are available")
        full = False
    else:
        raise FileNotFoundError("Full cohort is incomplete; pass --allow-partial only for interim audit")

    if list(sequences) != [seq for seq in protocol["sequences"] if seq in set(sequences)]:
        raise ValueError("Candidate sequences are not in frozen protocol order")
    if full and sequences != protocol["sequences"]:
        raise ValueError("Full result sequence list differs from the frozen 21-sequence cohort")

    manifest = load_manifest(args.manifest)
    yolo_by_sequence = {seq: [] for seq in sequences}
    for row in baseline_scores["frames"]:
        if row.get("post_prompt") is True and row["sequence"] in yolo_by_sequence:
            yolo_by_sequence[row["sequence"]].append(row)

    per_sequence: dict[str, Any] = {}
    candidate_digest_format_checks = 0
    yolo_hash_checks = 0
    gt_hash_checks = 0
    identity_checks = 0

    for seq in sequences:
        rows_by_condition = {
            condition: read_json(run_dir / condition / seq / "frames.json")
            for condition in CONDITIONS
        }
        frame_ids = {
            condition: [int(row["frame"]) for row in rows]
            for condition, rows in rows_by_condition.items()
        }
        if any(ids != frame_ids["noop"] for ids in frame_ids.values()):
            raise ValueError(f"{seq}: candidate condition frame IDs differ")

        yolo_rows = sorted(yolo_by_sequence[seq], key=lambda row: int(row["frame_index"]))
        yolo_ids = [int(row["frame_index"]) for row in yolo_rows]
        if yolo_ids != frame_ids["noop"]:
            raise ValueError(f"{seq}: saved YOLO and SAM2 frame IDs differ")
        expected = protocol["sequence_targets"][seq]
        if len(yolo_rows) != int(expected["targets"]):
            raise ValueError(f"{seq}: matched target count differs from the frozen protocol")

        yolo_prop_rows = []
        yolo_present_dice = []
        yolo_present_bf = []
        for target_index, yolo_row in zip(yolo_ids, yolo_rows):
            manifest_row = manifest[seq][target_index]
            if Path(manifest_row["image"]).stem != yolo_row["frame"]:
                raise ValueError(f"{seq}/{target_index}: manifest frame identity differs")
            gt_path = args.dataset_root / manifest_row["mask"]
            pred_path = args.baseline_output_dir / "masks" / seq / f"{yolo_row['frame']}.png"
            if sha256(gt_path) != yolo_row["gt_sha256"]:
                raise ValueError(f"{seq}/{target_index}: current GT hash mismatch")
            gt_hash_checks += 1
            if sha256(pred_path) != yolo_row["prediction_sha256"]:
                raise ValueError(f"{seq}/{target_index}: saved YOLO-mask hash mismatch")
            yolo_hash_checks += 1

            gt = load_label_mask(gt_path, {1}) == 1
            prediction = load_label_mask(pred_path, {1}) == 1
            present = bool(gt.any())
            if present != bool(yolo_row["gt_present"]):
                raise ValueError(f"{seq}/{target_index}: current GT presence differs")
            score = calculate_scores(prediction, gt)
            if abs(score["dice"] - float(yolo_row["recomputed_dice"])) > 1e-8:
                raise ValueError(f"{seq}/{target_index}: saved YOLO Dice does not reproduce")
            if abs(score["dice"] - float(yolo_row["recorded_dice"])) > 1e-8:
                raise ValueError(f"{seq}/{target_index}: recorded YOLO Dice does not reproduce")
            if abs(score["iou"] - float(yolo_row["recomputed_iou"])) > 1e-8:
                raise ValueError(f"{seq}/{target_index}: saved YOLO IoU does not reproduce")

            row_false_alarm = bool(prediction.any()) if not present else None
            if not present and row_false_alarm != bool(yolo_row["false_alarm"]):
                raise ValueError(f"{seq}/{target_index}: saved YOLO false-alarm label differs")
            if present:
                yolo_present_dice.append(float(score["dice"]))
                yolo_present_bf.append(float(db_eval_boundary(prediction, gt)))
            yolo_prop_rows.append({
                "frame": int(target_index),
                "present": present,
                "scores": score,
                "absent_fp": row_false_alarm,
            })
            for condition in CONDITIONS:
                candidate_row = rows_by_condition[condition][len(yolo_prop_rows) - 1]
                if bool(candidate_row["present"]) != present:
                    raise ValueError(f"{seq}/{target_index}: candidate GT-presence label differs")
                candidate_digest = candidate_row.get("prediction_sha256")
                if not isinstance(candidate_digest, str) or re.fullmatch(
                    r"[0-9a-f]{64}", candidate_digest
                ) is None:
                    raise ValueError(
                        f"{seq}/{target_index}/{condition}: missing or malformed prediction digest"
                    )
                candidate_digest_format_checks += 1
            identity_checks += 1

        yolo_summary = {
            "targets": len(yolo_rows),
            "present_targets": len(yolo_present_dice),
            "empty_targets": len(yolo_rows) - len(yolo_present_dice),
            "present_dice": mean(yolo_present_dice),
            "present_boundary_f": mean(yolo_present_bf),
            "false_alarms": sum(bool(row["absent_fp"]) for row in yolo_prop_rows if not row["present"]),
            **{
                key: int(propagation_summary(yolo_prop_rows)[key])
                for key in ("return_count", "undetected_returns")
            },
        }
        if yolo_summary["present_targets"] != int(expected["present_targets"]):
            raise ValueError(f"{seq}: present-frame denominator differs from protocol")
        if yolo_summary["empty_targets"] != int(expected["gt_empty_targets"]):
            raise ValueError(f"{seq}: GT-empty denominator differs from protocol")
        if abs(
            yolo_summary["present_dice"]
            - float(baseline_reaggregation["by_sequence"][seq]["post_prompt_present_dice"])
        ) > 1e-12:
            raise ValueError(f"{seq}: per-sequence YOLO aggregation does not reproduce")

        per_sequence[seq] = {"framewise_yolo": yolo_summary}
        # The baseline mask audit's recomputed rows are the authoritative values.
        yolo_scores = [float(row["recomputed_dice"]) for row in yolo_rows if row["gt_present"]]
        if abs(mean(yolo_scores) - yolo_summary["present_dice"]) > 1e-12:
            raise ValueError(f"{seq}: recomputed YOLO sequence mean differs")

        for condition in CONDITIONS:
            candidate_rows = rows_by_condition[condition]
            summary = _summarize_rows(candidate_rows)
            logged = summaries[condition][seq]
            for metric in ("present_dice", "present_boundary_f", "false_alarms", "return_count", "undetected_returns"):
                logged_key = "boundary_f" if metric == "present_boundary_f" else (
                    "gt_empty_false_alarm_count" if metric == "false_alarms" else metric
                )
                if logged_key in logged and abs(float(summary[metric]) - float(logged[logged_key])) > 1e-10:
                    raise ValueError(f"{seq}/{condition}: frame rows disagree with logged {logged_key}")
            per_sequence[seq][condition] = summary

    if full:
        if identity_checks != int(protocol["unique_post_prompt_targets"]):
            raise ValueError("Full comparison did not match every post-prompt target")
        if sum(row["framewise_yolo"]["present_targets"] for row in per_sequence.values()) != int(baseline_reaggregation["post_prompt_present_frames"]):
            raise ValueError("Full post-prompt present-frame denominator differs from baseline audit")
        if abs(
            mean([row["framewise_yolo"]["present_dice"] for row in per_sequence.values()])
            - float(baseline_reaggregation["eligible21_sequence_macro_post_prompt_present"])
        ) > 1e-12:
            raise ValueError("Full frame-wise YOLO aggregation does not reproduce baseline reaggregation")

    means = {}
    for method in (*CONDITIONS, "framewise_yolo"):
        means[method] = {
            "present_dice": mean([per_sequence[seq][method]["present_dice"] for seq in sequences]),
            "present_boundary_f": mean([per_sequence[seq][method]["present_boundary_f"] for seq in sequences]),
            "false_alarms": sum(per_sequence[seq][method]["false_alarms"] for seq in sequences),
            "empty_targets": sum(per_sequence[seq][method]["empty_targets"] for seq in sequences),
            "return_count": sum(per_sequence[seq][method]["return_count"] for seq in sequences),
            "undetected_returns": sum(per_sequence[seq][method]["undetected_returns"] for seq in sequences),
        }

    contrasts = {}
    p_values = {}
    for condition in CONDITIONS:
        deltas = {
            seq: per_sequence[seq][condition]["present_dice"]
            - per_sequence[seq]["framewise_yolo"]["present_dice"]
            for seq in sequences
        }
        ci = sequence_bootstrap_interval(
            deltas,
            seed=int(protocol["statistics"]["bootstrap_seed"]),
            resamples=int(protocol["statistics"]["bootstrap_resamples"]),
            confidence=float(protocol["statistics"]["confidence"]),
        )
        test = paired_wilcoxon(
            {seq: per_sequence[seq][condition]["present_dice"] for seq in sequences},
            {seq: per_sequence[seq]["framewise_yolo"]["present_dice"] for seq in sequences},
        )
        p_values[condition] = test["p"]
        contrasts[condition] = {
            "mean_delta": mean(list(deltas.values())),
            "sequence_bootstrap_ci95": ci,
            "paired_wilcoxon": test,
            "wins_ties_losses": [
                sum(v > 1e-12 for v in deltas.values()),
                sum(abs(v) <= 1e-12 for v in deltas.values()),
                sum(v < -1e-12 for v in deltas.values()),
            ],
            "mean_delta_without_seq8": mean([v for seq, v in deltas.items() if seq != "seq8"])
            if len(deltas) > 1 and "seq8" in deltas else mean(list(deltas.values())),
        }
    adjusted = holm_adjust(p_values)
    for condition in CONDITIONS:
        contrasts[condition]["holm_adjusted_p_descriptive"] = adjusted[condition]

    runtime_file = run_dir / "runtime.json"
    runtimes = read_json(runtime_file) if runtime_file.exists() else results.get("runtimes", {})
    runtime_seconds = {}
    for condition in CONDITIONS:
        values = runtimes.get(condition, {})
        runtime_seconds[condition] = sum(
            float(receipt.get("wall_seconds", 0.0))
            for seq, receipt in values.items() if seq in sequences
        )
    runtime_seconds["framewise_yolo"] = None

    return {
        "status": "verified descriptive matched full-cohort saved-output comparison"
        if full else "verified descriptive interim matched-subset saved-output comparison",
        "inference_rerun": False,
        "inference_provenance_complete": False,
        "scope": "same post-prompt manifest frames and GT; per-sequence present-frame Dice, then equal sequence mean",
        "sequences": sequences,
        "n_sequences": len(sequences),
        "identity_checks_passed": identity_checks,
        "current_gt_hash_checks_passed": gt_hash_checks,
        "saved_yolo_mask_hash_checks_passed": yolo_hash_checks,
        "candidate_prediction_digest_format_checks_passed": candidate_digest_format_checks,
        "candidate_metric_verification": "Reaggregated saved per-frame score rows; candidate masks are not independently rescored by this audit.",
        "shared_evaluator_sha256": protocol["source_hashes"]["evaluation/segmentation.py"],
        "means": means,
        "paired_sequence_contrasts_vs_yolo": contrasts,
        "runtime_seconds": runtime_seconds,
        "candidate_settings": _candidate_settings(protocol),
        "framewise_yolo_saved_mask_shape": baseline_scores["frames"][0]["prediction_shape"],
        "per_sequence": per_sequence,
        "yolo_reference_aggregations": {
            "eligible21_all_frames": baseline_reaggregation["eligible21_sequence_macro_all_frames"],
            "eligible21_post_prompt_present": baseline_reaggregation["eligible21_sequence_macro_post_prompt_present"],
            "post_prompt_present_targets": baseline_reaggregation["post_prompt_present_frames"],
            "post_prompt_empty_targets": baseline_reaggregation["post_prompt_empty_frames"],
            "post_prompt_false_alarms": baseline_reaggregation["post_prompt_false_alarms_from_recorded_empty_dice"],
        },
        "input_sha256": {
            "candidate_setup": sha256(run_dir / "setup.json"),
            "candidate_result": sha256(results_path if full else run_dir / "partial_results.json"),
            "candidate_completion": sha256(completion_path) if full else None,
            "protocol": sha256(args.protocol),
            "manifest": sha256(args.manifest),
            "baseline_saved_mask_recomputation": sha256(baseline_audit / "saved_mask_recomputation.json"),
            "baseline_recorded_reaggregation": sha256(baseline_audit / "recorded_baseline_reaggregation.json"),
            "audit_script": sha256(Path(__file__)),
        },
        "provenance_gaps": [
            "Historical YOLO detector weights, confidence threshold, and box-selection policy are not recorded.",
            "Historical SAM2 checkpoint/source revision, preprocessing resolution, and prompt/box frequency are not recorded.",
            "Historical YOLO/SAM2 runtime and compute are unavailable.",
            "Therefore this is a same-frame saved-output comparison, not a fully matched system benchmark.",
        ],
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Cross-frame ToMe vs saved frame-wise YOLO-box masks",
        "",
        f"Status: {summary['status']}; {summary['n_sequences']} sequences, "
        f"{summary['identity_checks_passed']} exact frame joins. No inference was rerun.",
        "The YOLO comparison is descriptive because historical detector/checkpoint/prompt/runtime provenance is incomplete.",
        "",
        "## Equal-sequence mean present-frame metrics",
        "",
        "| Method | Dice | Δ Dice vs YOLO | Boundary F | False alarms / GT-empty frames | Returns | Undetected returns | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    yolo_dice = summary["means"]["framewise_yolo"]["present_dice"]
    for method in ("framewise_yolo", *CONDITIONS):
        row = summary["means"][method]
        delta = 0.0 if method == "framewise_yolo" else row["present_dice"] - yolo_dice
        runtime = summary["runtime_seconds"][method]
        lines.append(
            f"| {LABELS[method]} | {row['present_dice']:.6f} | "
            f"{'—' if method == 'framewise_yolo' else f'{delta:+.6f}'} | "
            f"{row['present_boundary_f']:.6f} | {row['false_alarms']} / {row['empty_targets']} | "
            f"{row['return_count']} | {row['undetected_returns']} | "
            f"{'unavailable' if runtime is None else f'{runtime:.1f}'} |"
        )
    lines.extend([
        "",
        "## Paired sequence uncertainty vs YOLO",
        "",
        "Intervals and tests use sequences as the sampling units. These are descriptive because the historic YOLO run is not a fully matched protocol.",
        "",
        "| Method | Mean paired Δ Dice | Sequence-bootstrap 95% CI | Wilcoxon p | Holm p (descriptive) | Wins / ties / losses | Δ without seq8 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        row = summary["paired_sequence_contrasts_vs_yolo"][condition]
        ci = row["sequence_bootstrap_ci95"]
        lines.append(
            f"| {LABELS[condition]} | {row['mean_delta']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | "
            f"{row['paired_wilcoxon']['p']:.4f} | {row['holm_adjusted_p_descriptive']:.4f} | "
            f"{row['wins_ties_losses'][0]} / {row['wins_ties_losses'][1]} / {row['wins_ties_losses'][2]} | "
            f"{row['mean_delta_without_seq8']:+.6f} |"
        )
    lines.extend([
        "",
        "## Per-sequence present-frame Dice",
        "",
        "| Sequence | Present / empty frames | YOLO | Native | Within-frame control | Cross-frame ToMe |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for seq, row in summary["per_sequence"].items():
        yolo = row["framewise_yolo"]
        lines.append(
            f"| {seq} | {yolo['present_targets']} / {yolo['empty_targets']} | "
            f"{yolo['present_dice']:.6f} | {row['noop']['present_dice']:.6f} | "
            f"{row['local']['present_dice']:.6f} | {row['cross_frame']['present_dice']:.6f} |"
        )
    baseline = summary["yolo_reference_aggregations"]
    lines.extend([
        "",
        "## Denominator and interpretation",
        "",
        f"The saved YOLO-mask audit reports {float(baseline['eligible21_all_frames']):.6f} for eligible21 all-frame Dice and "
        f"{float(baseline['eligible21_post_prompt_present']):.6f} for equal-sequence post-prompt GT-present Dice. "
        f"The full 21-sequence reference contains {baseline['post_prompt_present_targets']} present and "
        f"{baseline['post_prompt_empty_targets']} GT-empty post-prompt frames. Do not compare the all-frame score as if it used the present-frame denominator.",
        "",
        "YOLO inference provenance remains incomplete, so score gaps do not establish a fair system-level ranking. Runtime is unavailable for the saved YOLO run.",
        "Candidate metrics are reaggregated from saved per-frame score rows; candidate masks are not independently rescored by this audit. YOLO masks are independently rescored.",
        "",
        "## Recorded and missing system settings",
        "",
        f"SAM2 candidate: checkpoint SHA256 `{summary['candidate_settings']['sam2_checkpoint_sha256']}`, "
        f"reference revision `{summary['candidate_settings']['sam2_reference_revision']}`, "
        f"image size {summary['candidate_settings']['sam2_image_size']}, CPU/"
        f"{summary['candidate_settings']['runtime']['dtype']} with "
        f"{summary['candidate_settings']['runtime']['threads']} threads. Prompt policy: "
        f"{summary['candidate_settings']['prompt_policy']}. Prompt frame indices are recorded in the JSON report.",
        f"YOLO saved-mask shape: {summary['framewise_yolo_saved_mask_shape']}; this is output-mask shape, not detector input resolution. "
        "YOLO detector/checkpoint, preprocessing/input resolution, box cadence, and compute are unknown.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-audit-dir", type=Path, required=True)
    parser.add_argument("--baseline-output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary = audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "interim" if summary["n_sequences"] < 21 else "full"
    json_path = args.output_dir / f"spatiotemporal_yolo_matched_{suffix}.json"
    md_path = args.output_dir / f"spatiotemporal_yolo_matched_{suffix}.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    md_path.write_text(render(summary))
    print(render(summary))
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
