"""Summarize a completed Run174 result without rerunning inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "iterative_memory_pilot_v1.json"
ANALYSIS_PROTOCOL = ROOT / "protocols" / "iterative_memory_reporting_v1.json"
ARMS = ("native", "single_regional", "iterative_generic", "iterative_regional")
CONTROLS = ("native", "iterative_generic", "single_regional")
METRICS = (
    "present_dice",
    "present_iou",
    "present_boundary_f",
    "all_frame_dice",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _paired_dice(per_sequence, names, control, seed):
    candidate = per_sequence["iterative_regional"]
    baseline = per_sequence[control]
    paired_names = [
        name
        for name in names
        if finite_number(candidate[name].get("present_dice"))
        and finite_number(baseline[name].get("present_dice"))
    ]
    delta = {
        name: float(candidate[name]["present_dice"] - baseline[name]["present_dice"])
        for name in paired_names
    }
    loo = {}
    for omitted in paired_names:
        remaining = [name for name in paired_names if name != omitted]
        loo[omitted] = float(np.mean([delta[name] for name in remaining])) if remaining else None
    full_mean = float(np.mean(list(delta.values()))) if delta else None
    influence = {
        name: float(value - full_mean)
        for name, value in loo.items()
        if value is not None and full_mean is not None
    }
    test = paired_wilcoxon(
        {name: candidate[name]["present_dice"] for name in paired_names},
        {name: baseline[name]["present_dice"] for name in paired_names},
    )
    return {
        "n_sequences": len(paired_names),
        "missing_sequences": [name for name in names if name not in paired_names],
        "mean_delta": full_mean,
        "median_delta": float(np.median(list(delta.values()))) if delta else None,
        "sequence_deltas": delta,
        "wins": sum(value > 0 for value in delta.values()),
        "ties": sum(value == 0 for value in delta.values()),
        "losses": sum(value < 0 for value in delta.values()),
        "bootstrap_95_ci": sequence_bootstrap_interval(delta, seed=seed) if delta else None,
        "wilcoxon_p_unadjusted": test["p"],
        "leave_one_sequence_out_mean_deltas": loo,
        "leave_one_out_min": min(loo.values()) if loo else None,
        "leave_one_out_max": max(loo.values()) if loo else None,
        "max_absolute_leave_one_out_influence": max(map(abs, influence.values())) if influence else None,
        "leave_one_out_same_sign_count": (
            None if full_mean in (None, 0.0) else sum(
                (value > 0) == (full_mean > 0)
                for value in loo.values()
                if value is not None
            )
        ),
        "interpretation_limit": "Leave-one-sequence-out values are sensitivity summaries, not alternative test sets or grounds for excluding a sequence.",
    }


def analyze_results(results: dict, protocol: dict) -> dict:
    """Create sequence-level descriptive metrics and prespecified contrasts."""
    names = protocol["data"]["test"]
    per_sequence = results.get("per_sequence")
    if not isinstance(per_sequence, dict) or set(per_sequence) != set(ARMS):
        raise ValueError(f"Expected exactly these result arms: {', '.join(ARMS)}")
    for arm in ARMS:
        if set(per_sequence[arm]) != set(names):
            raise ValueError(f"{arm} result sequence set differs from frozen test cohort")

    arms = {}
    for arm in ARMS:
        records = per_sequence[arm]
        metric_summary = {}
        for metric in METRICS:
            values = {
                name: float(records[name][metric])
                for name in names
                if finite_number(records[name].get(metric))
            }
            metric_summary[metric] = {
                "mean_across_sequences": float(np.mean(list(values.values()))) if values else None,
                "median_across_sequences": float(np.median(list(values.values()))) if values else None,
                "n_sequences": len(values),
                "missing_sequences": [name for name in names if name not in values],
                "per_sequence": values,
            }

        empty_frames = sum(int(records[name].get("gt_empty_frames", 0)) for name in names)
        false_alarms = sum(int(records[name].get("gt_empty_false_alarm_count", 0)) for name in names)
        false_alarm_counts_by_sequence = {
            name: int(records[name].get("gt_empty_false_alarm_count", 0)) for name in names
        }
        empty_frames_by_sequence = {
            name: int(records[name].get("gt_empty_frames", 0)) for name in names
        }
        per_sequence_false_alarm_rates = {}
        total_targets = 0
        total_elapsed = 0.0
        sequence_fps = {}
        sequence_returns = {}
        all_events = []
        for name in names:
            row = records[name]
            denominator = int(row.get("gt_empty_frames", 0))
            numerator = int(row.get("gt_empty_false_alarm_count", 0))
            if denominator:
                per_sequence_false_alarm_rates[name] = numerator / denominator
            targets = int(row.get("target_frames", 0))
            elapsed = float(row.get("elapsed_seconds", 0.0))
            total_targets += targets
            total_elapsed += elapsed
            if elapsed > 0 and targets > 0:
                sequence_fps[name] = targets / elapsed
            propagation = row.get("propagation") or {}
            events = propagation.get("returns") or []
            detected = [event for event in events if event.get("detected") is True]
            delays = [int(event["delay_sampled_frames"]) for event in detected
                      if event.get("delay_sampled_frames") is not None]
            sequence_returns[name] = {
                "return_count": len(events),
                "detected_returns": len(detected),
                "undetected_or_censored_returns": len(events) - len(detected),
                "median_detected_delay_sampled_frames": float(np.median(delays)) if delays else None,
                "mean_detected_delay_sampled_frames": float(np.mean(delays)) if delays else None,
            }
            all_events.extend(events)

        pooled_detected = [event for event in all_events if event.get("detected") is True]
        pooled_delays = [int(event["delay_sampled_frames"]) for event in pooled_detected
                         if event.get("delay_sampled_frames") is not None]
        added_parameters = {
            "native": 0,
            "single_regional": int(protocol["parameters"]["single_regional"]),
            "iterative_generic": int(protocol["parameters"]["iterative"]),
            "iterative_regional": int(protocol["parameters"]["iterative"]),
        }[arm]
        arms[arm] = {
            "metrics": metric_summary,
            "false_alarms": {
                "count": false_alarms,
                "ground_truth_empty_frames": empty_frames,
                "pooled_rate": false_alarms / empty_frames if empty_frames else None,
                "macro_sequence_rate": float(np.mean(list(per_sequence_false_alarm_rates.values())))
                if per_sequence_false_alarm_rates else None,
                "sequences_with_empty_frames": len(per_sequence_false_alarm_rates),
                "per_sequence_count": false_alarm_counts_by_sequence,
                "per_sequence_empty_frames": empty_frames_by_sequence,
                "per_sequence_rate": per_sequence_false_alarm_rates,
            },
            "redetection": {
                "return_events": len(all_events),
                "detected_returns": len(pooled_detected),
                "undetected_or_censored_returns": len(all_events) - len(pooled_detected),
                "detected_return_rate": len(pooled_detected) / len(all_events) if all_events else None,
                "median_detected_delay_sampled_frames": float(np.median(pooled_delays)) if pooled_delays else None,
                "mean_detected_delay_sampled_frames": float(np.mean(pooled_delays)) if pooled_delays else None,
                "units": "sampled manifest frames; episodes censored at sequence end remain counted as undetected",
                "per_sequence": sequence_returns,
            },
            "compute": {
                "total_propagation_seconds": total_elapsed,
                "mean_seconds_per_sequence": total_elapsed / len(names),
                "aggregate_fps": total_targets / total_elapsed if total_elapsed else None,
                "mean_sequence_fps": float(np.mean(list(sequence_fps.values()))) if sequence_fps else None,
                "per_sequence_fps": sequence_fps,
                "added_reader_parameters": added_parameters,
            },
        }

    contrasts = {
        control: _paired_dice(per_sequence, names, control, protocol["seed"])
        for control in CONTROLS
    }
    adjusted = holm_adjust({
        control: contrast["wilcoxon_p_unadjusted"]
        for control, contrast in contrasts.items()
    })
    for control, value in adjusted.items():
        contrasts[control]["wilcoxon_p_holm_3"] = value

    return {
        "analysis": "Run174 predeclared supplementary sequence-level summary",
        "cohort": names,
        "unit": "sequence; no frames are treated as independent samples",
        "method": "Metrics and contrasts recomputed from the frozen per-sequence results; no inference or model selection is performed.",
        "arms": arms,
        "iterative_regional_present_dice_contrasts": contrasts,
        "outlier_sensitivity": "All leave-one-sequence-out means are reported. No sequence is removed from the primary result.",
        "limitations": [
            "PolypGen sequences were previously examined; this pilot is exploratory, not untouched confirmation.",
            "Re-detection delays use sampled manifest frames, not elapsed seconds or every source-video frame.",
            "Added-reader parameter counts are from the frozen protocol; elapsed/FPS values measure the recorded propagation interval.",
        ],
    }


def markdown_summary(analysis: dict) -> str:
    lines = [
        "# Run174 supplementary result summary",
        "",
        "This report is generated from the completed frozen `results.json`; it runs no inference and does not alter the continuation decision.",
        "",
        "## Per-arm results",
        "",
        "| Arm | Present Dice | Present IoU | Boundary F | False alarms / empty frames | Re-detection (detected / events) | Median delay (sampled frames) | Propagation sec | FPS | Added reader params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in analysis["arms"].items():
        metric = row["metrics"]

        def fmt(value):
            return "NA" if value is None else f"{value:.4f}"

        false_alarms = row["false_alarms"]
        redetection = row["redetection"]
        compute = row["compute"]
        lines.append(
            f"| {arm} | {fmt(metric['present_dice']['mean_across_sequences'])} "
            f"| {fmt(metric['present_iou']['mean_across_sequences'])} "
            f"| {fmt(metric['present_boundary_f']['mean_across_sequences'])} "
            f"| {false_alarms['count']} / {false_alarms['ground_truth_empty_frames']} "
            f"| {redetection['detected_returns']} / {redetection['return_events']} "
            f"| {fmt(redetection['median_detected_delay_sampled_frames'])} "
            f"| {compute['total_propagation_seconds']:.1f} "
            f"| {fmt(compute['aggregate_fps'])} "
            f"| {compute['added_reader_parameters']} |"
        )
    lines += [
        "",
        "## Paired present-Dice contrasts: iterative regional minus control",
        "",
        "| Control | n | Mean Δ | Median Δ | 95% sequence-bootstrap CI | Wins/ties/losses | Holm p | Leave-one-out mean range | Max absolute leave-one-out influence |",
        "|---|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for control, row in analysis["iterative_regional_present_dice_contrasts"].items():
        ci = row["bootstrap_95_ci"]
        ci_text = "NA" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        loo_min, loo_max = row["leave_one_out_min"], row["leave_one_out_max"]
        loo_text = "NA" if loo_min is None else f"[{loo_min:+.4f}, {loo_max:+.4f}]"
        mean = row["mean_delta"]
        median = row["median_delta"]
        lines.append(
            f"| {control} | {row['n_sequences']} | {'NA' if mean is None else f'{mean:+.4f}'} "
            f"| {'NA' if median is None else f'{median:+.4f}'} | {ci_text} "
            f"| {row['wins']}/{row['ties']}/{row['losses']} "
            f"| {row['wilcoxon_p_holm_3']:.4g} | {loo_text} "
            f"| {fmt(row['max_absolute_leave_one_out_influence'])} |"
        )
    lines += [
        "",
        "## Per-sequence diagnostics",
        "",
        "| Sequence | Arm | Present Dice | Boundary F | False alarms / GT-empty frames | Re-detected / returns | Median delay (sampled frames) | FPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = analysis["cohort"]
    for name in names:
        for arm, row in analysis["arms"].items():
            dice = row["metrics"]["present_dice"]["per_sequence"].get(name)
            boundary = row["metrics"]["present_boundary_f"]["per_sequence"].get(name)
            false_alarms = row["false_alarms"]
            denominator = false_alarms["per_sequence_empty_frames"][name]
            false_alarm_cell = f"{false_alarms['per_sequence_count'][name]} / {denominator}"
            redetection = row["redetection"]["per_sequence"][name]
            delay = redetection["median_detected_delay_sampled_frames"]
            compute = row["compute"]["per_sequence_fps"].get(name)
            fmt = lambda value: "NA" if value is None else f"{value:.4f}"
            lines.append(
                f"| {name} | {arm} | {fmt(dice)} | {fmt(boundary)} "
                f"| {false_alarm_cell} | {redetection['detected_returns']}/{redetection['return_count']} "
                f"| {fmt(delay)} | {fmt(compute)} |"
            )
    lines += ["", "## Limits", ""]
    lines += [f"- {item}" for item in analysis["limitations"]]
    lines += ["", "The leave-one-sequence-out table is a sensitivity check. It does not authorize dropping a sequence or changing the frozen continuation rule.", ""]
    return "\n".join(lines)


def write_report(
    results_path: Path,
    output_dir: Path,
    protocol_path: Path = PROTOCOL,
    analysis_protocol_path: Path = ANALYSIS_PROTOCOL,
) -> dict:
    results = json.loads(results_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    analysis_protocol = json.loads(analysis_protocol_path.read_text())
    implementation = analysis_protocol["implementation"]
    actual_script_sha = digest(Path(__file__))
    actual_test_sha = digest(ROOT / implementation["test"])
    if actual_script_sha != implementation["script_sha256"]:
        raise ValueError("Supplementary summarizer differs from its frozen source hash")
    if actual_test_sha != implementation["test_sha256"]:
        raise ValueError("Supplementary summarizer test differs from its frozen source hash")
    analysis = analyze_results(results, protocol)
    analysis["input_results_sha256"] = digest(results_path)
    analysis["protocol_sha256"] = digest(protocol_path)
    analysis["analysis_protocol_sha256"] = digest(analysis_protocol_path)
    analysis["analysis_script_sha256"] = actual_script_sha
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis.json"
    md_path = output_dir / "analysis.md"
    json_tmp = json_path.with_suffix(".json.tmp")
    md_tmp = md_path.with_suffix(".md.tmp")
    json_tmp.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    md_tmp.write_text(markdown_summary(analysis))
    json_tmp.replace(json_path)
    md_tmp.replace(md_path)
    return {"analysis_json": str(json_path), "analysis_markdown": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--analysis-protocol", type=Path, default=ANALYSIS_PROTOCOL)
    args = parser.parse_args()
    print(json.dumps(write_report(
        args.results, args.output_dir, args.protocol, args.analysis_protocol
    ), indent=2))


if __name__ == "__main__":
    main()
