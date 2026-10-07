#!/usr/bin/env python3
"""Apply the pre-registered gain-selection rule on the dev split.

For each run, compute per-sequence overall Dice, lost-polyp-frame fraction
(polyp frames with Dice < 0.1), and high-motion polyp-frame Dice (camera motion
above a fixed tercile cut). A candidate gain beats the plain Kalman memory only
if it lowers the lost fraction AND raises high-motion Dice. Camera motion is the
per-frame ``feature_change_mean`` logged in each run's diagnostics.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import wilcoxon


def load_frames(run_dir: Path):
    """Map (sequence, frame) -> row with dice, gt_present, motion."""
    per_frame = run_dir / "evaluation" / "metrics_per_frame.csv"
    motion = {}
    diag_dir = run_dir / "masks" / "diagnostics"
    for path in diag_dir.glob("seq*.csv"):
        for row in csv.DictReader(path.open()):
            if row.get("feature_change_mean"):
                motion[(path.stem, row["frame"])] = float(row["feature_change_mean"])
    rows = {}
    for row in csv.DictReader(per_frame.open()):
        if row["frames_after_prompt"] in ("", "nan"):
            continue
        key = (row["sequence"], row["frame"])
        rows[key] = {
            "sequence": row["sequence"],
            "dice": float(row["dice"]),
            "gt_present": row["gt_present"].lower() == "true",
            "motion": motion.get(key),
        }
    return rows


def per_sequence(rows, motion_cut):
    dice, lost, high = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows.values():
        s = r["sequence"]
        dice[s].append(r["dice"])
        if r["gt_present"]:
            lost[s].append(1.0 if r["dice"] < 0.1 else 0.0)
            if r["motion"] is not None and r["motion"] > motion_cut:
                high[s].append(r["dice"])
    seqs = sorted(dice, key=lambda x: int(x[3:]))
    return (
        {s: mean(dice[s]) for s in seqs},
        {s: mean(lost[s]) for s in seqs if lost[s]},
        {s: mean(high[s]) for s in seqs if high[s]},
    )


def paired(a, b):
    shared = sorted(a.keys() & b.keys(), key=lambda x: int(x[3:]))
    diffs = np.array([b[s] - a[s] for s in shared])
    if len(diffs) < 3:
        return None
    try:
        p = wilcoxon(diffs).pvalue
    except ValueError:
        p = float("nan")
    return {"n": len(diffs), "mean": float(diffs.mean()),
            "better": int((diffs > 0).sum()), "worse": int((diffs < 0).sum()), "p": float(p)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, default=Path.home() / "Library/Caches/adseg_work/exp_soft_gain")
    parser.add_argument("--motion_cut", type=float, default=0.117, help="Pre-registered dev high-motion tercile cut.")
    parser.add_argument("--baseline", default="kalman_dev")
    parser.add_argument("--runs", nargs="+", default=["ema_dev", "soft_all_dev", "soft_present_dev"])
    args = parser.parse_args()

    base_rows = load_frames(args.exp_dir / args.baseline)
    base_dice, base_lost, base_high = per_sequence(base_rows, args.motion_cut)
    print(f"{'run':>16} {'Dice':>7} {'lost':>7} {'hiMot':>7}   vs Kalman: dDice / dLost / dHiMot  (rule pass?)")
    print(f"{args.baseline:>16} {mean(base_dice.values()):>7.4f} {mean(base_lost.values()):>7.4f} "
          f"{mean(base_high.values()):>7.4f}")
    for run in args.runs:
        rows = load_frames(args.exp_dir / run)
        d, l, h = per_sequence(rows, args.motion_cut)
        pd, pl, ph = paired(base_dice, d), paired(base_lost, l), paired(base_high, h)
        # Rule: lower lost fraction AND higher high-motion Dice (direction of the mean).
        passes = pl and ph and pl["mean"] < 0 and ph["mean"] > 0
        print(f"{run:>16} {mean(d.values()):>7.4f} {mean(l.values()):>7.4f} {mean(h.values()):>7.4f}   "
              f"dDice {pd['mean']:+.4f} (p={pd['p']:.2f}) / "
              f"dLost {pl['mean']:+.4f} (p={pl['p']:.2f}) / "
              f"dHiMot {ph['mean']:+.4f} (p={ph['p']:.2f})   "
              f"{'PASS' if passes else 'fail'}")


if __name__ == "__main__":
    main()
