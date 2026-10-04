#!/usr/bin/env python3
"""Collect every evaluated run under ``--root`` into CSV tables in ``<root>/summary``.

A run is any directory containing ``evaluation/metrics_per_frame.csv`` (written by
``evaluation/temporal.py``). Optional files: ``masks/diagnostics/*.csv`` and
``masks/efficiency.json`` (from ``scripts/infer.py``), ``setup.json`` and
``history.jsonl`` (from ``scripts/train_kalman.py``).

Tables:
  main_results.csv      one row per run, frame-pooled metrics with 95% sequence-bootstrap intervals
  per_sequence.csv      run x sequence metrics
  paired_tests.csv      each run vs each reference run, paired over sequences (Wilcoxon, bootstrap interval)
  failure_cases.csv     sequences where a run is clearly worse than the native baseline
  training_summary.csv  start/end training statistics and drift flags
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

SETUP_KEYS = ("absence_weight", "distill_weight", "clip_difficulty", "absence_probability", "seed", "steps")
SEQUENCE_METRICS = ("propagated_dice", "present_dice", "absent_fp_rate", "present_detection_rate")
LOWER_IS_BETTER = {"absent_fp_rate"}
BOOTSTRAP_METRICS = ("propagated_dice", "present_dice", "absent_fp_rate")
BOOTSTRAP_SAMPLES = 2000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def mean(values) -> float:
    values = [value for value in values if value is not None and not np.isnan(value)]
    return float(np.mean(values)) if values else float("nan")


def load_frames(run: Path) -> list[dict]:
    scores = {}
    for path in (run / "masks" / "diagnostics").glob("*.csv"):
        for row in read_csv(path):
            if row.get("object_score") not in (None, ""):
                scores[(path.stem, Path(row["frame"]).stem)] = float(row["object_score"])
    frames = []
    for row in read_csv(run / "evaluation" / "metrics_per_frame.csv"):
        offset = row.get("frames_after_prompt", "")
        if offset in ("", "None") or int(offset) < 1:
            continue
        since = row.get("frames_since_reappearance", "")
        absence = row.get("preceding_absence", "")
        frames.append({
            "sequence": row["sequence"],
            "present": row["gt_present"] == "True",
            "predicted": row["predicted_present"] == "True",
            "dice": float(row["dice"]),
            "since": None if since in ("", "None") else int(since),
            "absence": None if absence in ("", "None") else int(absence),
            "score": scores.get((row["sequence"], row["frame"])),
        })
    return frames


def auroc(frames: list[dict]) -> float:
    present = [f["score"] for f in frames if f["present"] and f["score"] is not None]
    absent = [f["score"] for f in frames if not f["present"] and f["score"] is not None]
    if not present or not absent:
        return float("nan")
    return float(mannwhitneyu(present, absent).statistic / (len(present) * len(absent)))


def frame_metrics(frames: list[dict]) -> dict:
    present = [f for f in frames if f["present"]]
    absent = [f for f in frames if not f["present"]]

    def reappeared(low, high, min_absence):
        return mean([
            f["dice"] for f in present
            if f["since"] is not None and low <= f["since"] <= high and f["absence"] >= min_absence
        ])

    return {
        "frames": len(frames),
        "present_frames": len(present),
        "absent_frames": len(absent),
        "propagated_dice": mean(f["dice"] for f in frames),
        "present_dice": mean(f["dice"] for f in present),
        "absent_dice": mean(f["dice"] for f in absent),
        "absent_fp_rate": mean(float(f["predicted"]) for f in absent),
        "present_detection_rate": mean(float(f["predicted"]) for f in present),
        "presence_auroc": auroc(frames),
        "reappear_first_after_ge1": reappeared(0, 0, 1),
        "reappear_first_after_ge7": reappeared(0, 0, 7),
        "reappear_1to4_after_ge7": reappeared(1, 4, 7),
        "reappear_ge10_after_ge1": reappeared(10, 10**9, 1),
    }


def bootstrap_intervals(frames: list[dict], seed: int = 0) -> dict:
    """95% intervals of frame-pooled metrics, resampling whole sequences."""
    rng = np.random.default_rng(seed)
    by_sequence = {}
    for f in frames:
        by_sequence.setdefault(f["sequence"], []).append(f)
    groups = list(by_sequence.values())
    draws = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [f for index in rng.integers(len(groups), size=len(groups)) for f in groups[index]]
        present = [f["dice"] for f in sample if f["present"]]
        absent = [float(f["predicted"]) for f in sample if not f["present"]]
        draws["propagated_dice"].append(mean(f["dice"] for f in sample))
        draws["present_dice"].append(mean(present))
        draws["absent_fp_rate"].append(mean(absent))
    result = {}
    for metric, values in draws.items():
        result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = np.nanpercentile(values, [2.5, 97.5]).tolist()
    return result


def sequence_metrics(frames: list[dict]) -> dict[str, dict]:
    result = {}
    for sequence in sorted({f["sequence"] for f in frames}, key=lambda name: int(name[3:])):
        rows = [f for f in frames if f["sequence"] == sequence]
        present = [f for f in rows if f["present"]]
        absent = [f for f in rows if not f["present"]]
        result[sequence] = {
            "c6": int(sequence[3:]) >= 16,
            "present_frames": len(present),
            "absent_frames": len(absent),
            "propagated_dice": mean(f["dice"] for f in rows),
            "present_dice": mean(f["dice"] for f in present),
            "absent_fp_rate": mean(float(f["predicted"]) for f in absent),
            "present_detection_rate": mean(float(f["predicted"]) for f in present),
        }
    return result


def efficiency(run: Path) -> dict:
    path = run / "masks" / "efficiency.json"
    if not path.is_file():
        return {}
    rows = [row for row in json.loads(path.read_text()) if row.get("fps")]
    return {
        "retained_memory_mib": mean(row["stored_spatial_memory_bytes"] / 2**20 for row in rows),
        "fps": mean(row["fps"] for row in rows),
    }


def training_summary(run: Path) -> dict | None:
    history = run / "history.jsonl"
    if not history.is_file():
        return None
    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    if not rows:
        return {"run": run.name, "clips": 0}
    window = max(1, min(100, len(rows) // 5))
    summary = {"run": run.name, "clips": len(rows)}
    for key in ("loss", "segmentation", "presence", "absence", "distill", "dice_present",
                "false_positive_rate_absent", "gain_present", "gain_absent"):
        start = mean(row.get(key) for row in rows[:window])
        end = mean(row.get(key) for row in rows[-window:])
        summary[f"{key}_start"], summary[f"{key}_end"] = start, end
    summary["flag_gain_absent_above_0.8"] = summary["gain_absent_end"] > 0.8
    summary["flag_gain_gap_below_0.1"] = summary["gain_present_end"] - summary["gain_absent_end"] < 0.1
    return summary


def paired(run, reference, per_sequence) -> list[dict]:
    rows = []
    for metric in SEQUENCE_METRICS:
        a, b = per_sequence[run], per_sequence[reference]
        keys = [s for s in a if s in b and not np.isnan(a[s][metric]) and not np.isnan(b[s][metric])]
        x = np.array([a[s][metric] for s in keys])
        y = np.array([b[s][metric] for s in keys])
        draws = np.random.default_rng(0).integers(len(x), size=(BOOTSTRAP_SAMPLES, len(x))) if len(x) else None
        low, high = np.percentile((x - y)[draws].mean(axis=1), [2.5, 97.5]) if len(x) else (np.nan, np.nan)
        better = (x < y) if metric in LOWER_IS_BETTER else (x > y)
        worse = (x > y) if metric in LOWER_IS_BETTER else (x < y)
        rows.append({
            "run": run,
            "reference": reference,
            "metric": metric,
            "sequences": len(keys),
            "mean_run": float(x.mean()) if len(x) else float("nan"),
            "mean_reference": float(y.mean()) if len(y) else float("nan"),
            "mean_diff": float((x - y).mean()) if len(x) else float("nan"),
            "diff_ci_low": float(low),
            "diff_ci_high": float(high),
            "better": int(better.sum()),
            "worse": int(worse.sum()),
            "ties": int((x == y).sum()),
            "wilcoxon_p": float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--references", nargs="*", default=["native_gtbox", "kalman_untrained"])
    parser.add_argument("--dice_drop", type=float, default=0.05, help="Failure if propagated Dice drops by more.")
    parser.add_argument("--fp_rise", type=float, default=0.2, help="Failure if empty-frame FP rate rises by more.")
    args = parser.parse_args()

    runs = sorted(path.parent.parent for path in args.root.glob("*/evaluation/metrics_per_frame.csv"))
    if not runs:
        raise SystemExit(f"No evaluated runs under {args.root}")
    output = args.root / "summary"
    output.mkdir(exist_ok=True)

    main_rows, sequence_rows, per_sequence, training_rows = [], [], {}, []
    for run in runs:
        frames = load_frames(run)
        setup = json.loads((run / "setup.json").read_text()) if (run / "setup.json").is_file() else {}
        main_rows.append({
            "run": run.name,
            **{key: setup.get(key, "") for key in SETUP_KEYS},
            **frame_metrics(frames),
            **bootstrap_intervals(frames),
            **efficiency(run),
        })
        per_sequence[run.name] = sequence_metrics(frames)
        sequence_rows += [{"run": run.name, "sequence": s, **m} for s, m in per_sequence[run.name].items()]
        summary = training_summary(run)
        if summary:
            training_rows.append(summary)

    tests, failures = [], []
    references = [name for name in args.references if name in per_sequence]
    for run in per_sequence:
        for reference in references:
            if run != reference:
                tests += paired(run, reference, per_sequence)
    if "native_gtbox" in per_sequence:
        native = per_sequence["native_gtbox"]
        for run, sequences in per_sequence.items():
            if run == "native_gtbox":
                continue
            for sequence, metrics in sequences.items():
                base = native.get(sequence)
                if base is None:
                    continue
                dice_delta = metrics["propagated_dice"] - base["propagated_dice"]
                fp_delta = metrics["absent_fp_rate"] - base["absent_fp_rate"]
                if dice_delta <= -args.dice_drop or (not np.isnan(fp_delta) and fp_delta >= args.fp_rise):
                    failures.append({
                        "run": run,
                        "sequence": sequence,
                        "native_propagated_dice": base["propagated_dice"],
                        "run_propagated_dice": metrics["propagated_dice"],
                        "dice_delta": dice_delta,
                        "native_absent_fp_rate": base["absent_fp_rate"],
                        "run_absent_fp_rate": metrics["absent_fp_rate"],
                        "fp_delta": fp_delta,
                    })

    write_csv(output / "main_results.csv", main_rows)
    write_csv(output / "per_sequence.csv", sequence_rows)
    if tests:
        write_csv(output / "paired_tests.csv", tests)
    write_csv(output / "failure_cases.csv", failures or [{"run": "", "sequence": ""}])
    if training_rows:
        write_csv(output / "training_summary.csv", training_rows)


if __name__ == "__main__":
    main()
