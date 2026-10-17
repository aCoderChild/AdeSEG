#!/usr/bin/env python3
"""Analyze E3 attention-utility agreement across memory corruption conditions."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.utility_retrieval import paired_wilcoxon, sequence_bootstrap_interval


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sequence_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video"]].append(row["attention_spearman"])
    return {name: float(np.nanmean(values)) for name, values in grouped.items()}


def corruption_trend(conditions):
    names = ("ground_truth", "corruption_0.25", "corruption_0.50",
             "corruption_0.75", "corruption_1.00")
    if not all(name in conditions for name in names):
        return None
    sequences = set.intersection(*(set(conditions[name]) for name in names))
    rows = [
        {"sequence": sequence, "corruption_level": level,
         "rho": conditions[name][sequence]}
        for sequence in sorted(sequences)
        for level, name in enumerate(names)
        if np.isfinite(conditions[name][sequence])
    ]
    if len(sequences) < 2 or len(rows) < 10:
        return None
    result = MixedLM.from_formula(
        "rho ~ corruption_level", groups="sequence", data=pd.DataFrame(rows)
    ).fit(reml=False)
    return {"sequence_count": len(sequences), "observations": len(rows),
            "slope": float(result.params["corruption_level"]),
            "p_two_sided": float(result.pvalues["corruption_level"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", nargs=2, metavar=("NAME", "FRAMES_JSONL"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conditions = {name: sequence_means(load(Path(path))) for name, path in args.condition}
    if "predicted" not in conditions or "ground_truth" not in conditions:
        raise ValueError("E3 requires predicted and ground_truth conditions.")
    comparison = paired_wilcoxon(conditions["predicted"], conditions["ground_truth"])
    corruption_names = sorted(
        [name for name in conditions if name.startswith("corruption_")],
        key=lambda name: float(name.removeprefix("corruption_")),
    )
    corruption_means = [float(np.nanmean(list(conditions[name].values()))) for name in corruption_names]
    monotonic = all(left >= right for left, right in zip(corruption_means, corruption_means[1:]))
    trend = corruption_trend(conditions)
    summary = {
        "mean_spearman": {name: float(np.nanmean(list(values.values()))) for name, values in conditions.items()},
        "bootstrap_95ci": {name: sequence_bootstrap_interval(values) for name, values in conditions.items()},
        "predicted_vs_ground_truth": comparison,
        "corruption_order": corruption_names,
        "corruption_monotonic_nonincreasing": monotonic,
        "corruption_mixed_model": trend,
        # A valid E3 decision additionally needs protocol-compliant utility,
        # foreground saliency and the frozen target-frame sampling plan.
        "E3_pass": None,
        "exploratory_clean_vs_predicted_difference": bool(
            comparison["p"] < 0.05
            and summary_value(conditions["predicted"]) < summary_value(conditions["ground_truth"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "per_sequence": conditions}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def summary_value(values):
    return float(np.nanmean(list(values.values())))


if __name__ == "__main__":
    main()
