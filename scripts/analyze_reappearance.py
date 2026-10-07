#!/usr/bin/env python3
"""Reappearance-stratified Dice and empty-frame FP across methods.

Uses per-frame CSVs produced by evaluation/temporal.py and the
``frames_since_reappearance`` / ``preceding_absence`` columns.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import wilcoxon


DEV_SEQUENCES = {f"seq{i}" for i in range(1, 16)}
TEST_SEQUENCES = {f"seq{i}" for i in range(16, 24)}


def load_rows(path: Path):
    with path.open() as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def as_float(value: str) -> float:
    return float(value) if value not in ("", "nan") else float("nan")


def as_int(value: str):
    return int(value) if value not in ("", "nan") else None


def summarize(rows, bucket, keep):
    """Return mean Dice over frames the bucket function selects, grouped by sequence."""
    per_sequence = defaultdict(list)
    for row in rows:
        if not keep(row):
            continue
        key = bucket(row)
        if key is None:
            continue
        per_sequence[(row["sequence"], key)].append(as_float(row["dice"]))
    out = defaultdict(dict)
    for (seq, key), values in per_sequence.items():
        out[key][seq] = mean(values)
    return out


def paired(diff_by_sequence):
    if len(diff_by_sequence) < 3:
        return None
    values = np.asarray(list(diff_by_sequence.values()))
    try:
        _, p = wilcoxon(values)
    except ValueError:
        p = float("nan")
    return {"n": len(values), "mean": float(values.mean()), "p": float(p),
            "better": int((values > 0).sum()), "worse": int((values < 0).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", type=Path, required=True,
                        help="folder with native, native_prompt_plus_1frame, kalman and ema runs (outputs/polypgen/all23)")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min_preceding", type=int, default=1,
                        help="Only count reappearances after at least this many absent frames.")
    args = parser.parse_args()

    sequences = {"dev": DEV_SEQUENCES, "test": TEST_SEQUENCES,
                 "all": DEV_SEQUENCES | TEST_SEQUENCES}[args.split]
    runs = {
        "native": "native/evaluation/metrics_per_frame.csv",
        "native_k1": "native_prompt_plus_1frame/evaluation/metrics_per_frame.csv",
        "kalman": "kalman/evaluation/metrics_per_frame.csv",
        "ema": "ema/evaluation/metrics_per_frame.csv",
    }

    def present_bucket(row):
        offset = as_int(row["frames_since_reappearance"])
        prec = as_int(row["preceding_absence"])
        if offset is None or prec is None or prec < args.min_preceding:
            return None
        if offset > args.window:
            return None
        return offset

    def keep_polyp(row):
        return row["sequence"] in sequences and row["gt_present"].lower() == "true"

    def keep_empty(row):
        return row["sequence"] in sequences and row["gt_present"].lower() == "false"

    def post_prompt(row):
        after = as_int(row["frames_after_prompt"])
        return after is not None and after >= 0

    buckets = {}
    for name, filename in runs.items():
        rows = load_rows(args.analysis_dir / filename)
        rows = [r for r in rows if post_prompt(r)]
        buckets[name] = summarize(rows, present_bucket, keep_polyp)

    print(f"Reappearance Dice on polyp frames, {args.split} split")
    print(f"Precedent absence >= {args.min_preceding}, window = 1..{args.window}\n")
    header = ["offset"] + list(runs)
    print("\t".join(f"{h:>10}" for h in header))
    for offset in range(1, args.window + 1):
        line = [f"{offset:>10}"]
        for name in runs:
            seq_values = buckets[name].get(offset, {})
            m = mean(seq_values.values()) if seq_values else float("nan")
            line.append(f"{m:>10.3f}")
        print("\t".join(line))

    print("\nPaired tests (Kalman vs native) per offset:")
    for offset in range(1, args.window + 1):
        native = buckets["native"].get(offset, {})
        kalman = buckets["kalman"].get(offset, {})
        shared = native.keys() & kalman.keys()
        diffs = {s: kalman[s] - native[s] for s in shared}
        res = paired(diffs)
        if res is None:
            continue
        print(f"  offset {offset}: n={res['n']}  mean diff {res['mean']:+.3f}  "
              f"better/worse {res['better']}/{res['worse']}  p={res['p']:.3f}")

    print("\nAggregate (within window, averaged per sequence first):")
    agg = defaultdict(dict)
    for name, by_offset in buckets.items():
        for offset, seq_values in by_offset.items():
            for seq, value in seq_values.items():
                agg[name].setdefault(seq, []).append(value)
    for name in runs:
        seq_means = {seq: mean(vs) for seq, vs in agg[name].items()}
        if seq_means:
            print(f"  {name:>12}: n={len(seq_means)}  Dice {mean(seq_means.values()):.3f}")
    for other in ("kalman", "ema", "native_k1"):
        shared = agg["native"].keys() & agg[other].keys()
        diffs = {s: mean(agg[other][s]) - mean(agg["native"][s]) for s in shared}
        res = paired(diffs)
        if res:
            print(f"  {other:>12} vs native: n={res['n']}  diff {res['mean']:+.3f}  "
                  f"{res['better']}/{res['worse']}  p={res['p']:.3f}")

    out = args.analysis_dir / f"reappearance_{args.split}.json"
    out.write_text(json.dumps({name: {str(k): v for k, v in b.items()} for name, b in buckets.items()}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
