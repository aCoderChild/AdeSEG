#!/usr/bin/env python3
"""Classify E1 failure onsets from matched predicted- and GT-memory diagnostics."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.utility_retrieval import recovered_fraction


def read_rows(path):
    return {(row["video"], row["frame"]): row for row in map(json.loads, path.read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--ground-truth-memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    predicted, ground_truth = read_rows(args.predicted), read_rows(args.ground_truth_memory)
    if set(predicted) != set(ground_truth):
        raise ValueError("Predicted- and GT-memory diagnostics do not cover identical frames.")
    rows = []
    for key in sorted(predicted):
        normal, gt = predicted[key], ground_truth[key]
        prompt_fixes = normal["oracle_prompt_dice"] >= args.threshold
        memory_fixes = gt["baseline_dice"] >= args.threshold
        retrieval_fixes = normal["oracle_retrieval_dice"] >= args.threshold
        if not prompt_fixes:
            category = "image_limited"
        elif memory_fixes and retrieval_fixes:
            category = "both"
        elif memory_fixes:
            category = "construction"
        elif retrieval_fixes:
            category = "retrieval"
        else:
            category = "presence_or_unresolved"
        rows.append({
            "video": key[0], "frame": key[1], "category": category,
            "baseline_dice": normal["baseline_dice"],
            "oracle_prompt_dice": normal["oracle_prompt_dice"],
            "oracle_memory_dice": gt["baseline_dice"],
            "oracle_retrieval_dice": normal["oracle_retrieval_dice"],
            "best_of_n_random_dice": normal["best_of_n_random_dice"],
            "recovered_fraction": recovered_fraction(
                normal["oracle_retrieval_dice"] - normal["baseline_dice"],
                normal["best_of_n_random_dice"] - normal["baseline_dice"],
                gt["baseline_dice"] - normal["baseline_dice"],
            ),
        })
    counts = Counter(row["category"] for row in rows)
    total = len(rows)
    summary = {
        "episodes": total,
        "counts": dict(counts),
        "fractions": {name: count / total for name, count in counts.items()} if total else {},
        "retrieval_or_both_fraction": (
            counts["retrieval"] + counts["both"]
        ) / total if total else None,
        "mean_recovered_fraction": float(np.mean([
            row["recovered_fraction"] for row in rows if math.isfinite(row["recovered_fraction"])
        ])) if any(math.isfinite(row["recovered_fraction"]) for row in rows) else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
