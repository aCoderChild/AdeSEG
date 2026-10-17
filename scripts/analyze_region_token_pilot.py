#!/usr/bin/env python3
"""Evaluate the preregistered E1 gate using sequence averages, never frame n."""
# ruff: noqa: E402 -- allow direct CLI execution from any working directory

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.utility_retrieval import sequence_bootstrap_interval, rank_agreement
from modeling.region_tokens import CATEGORIES


def analyze(rows, required_sequences):
    complete = [row for row in rows if row["state"] == "complete"]
    sequences = sorted({row["video"] for row in complete})
    groups = defaultdict(list)
    for row in complete:
        for metric in ("dice", "boundary_f"):
            groups[
                (row["bank"], row["category"], row["seed"], row["video"], metric)
            ].append(row["harm"][metric])
    sequence_means = {key: float(np.mean(value)) for key, value in groups.items()}
    effects, gate_effect = {}, False
    for bank in ("conditioning", "recent"):
        for metric, minimum in (("dice", 0.01), ("boundary_f", 0.02)):
            for category in ("boundary", "near_background"):
                # Same sequences across all five random controls and category.
                names = [
                    video
                    for video in sequences
                    if (bank, category, 0, video, metric) in sequence_means
                    and all(
                        (bank, "random", seed, video, metric) in sequence_means
                        for seed in range(5)
                    )
                ]
                if not names:
                    continue
                by_sequence = {
                    video: sequence_means[(bank, category, 0, video, metric)]
                    for video in names
                }
                random_means = [
                    float(
                        np.mean(
                            [
                                sequence_means[(bank, "random", seed, video, metric)]
                                for video in names
                            ]
                        )
                    )
                    for seed in range(5)
                ]
                mean = float(np.mean(list(by_sequence.values())))
                passes = len(names) >= 8 and mean - max(random_means) >= minimum
                gate_effect |= passes
                effects[f"{bank}/{category}/{metric}"] = {
                    "n_sequences": len(names),
                    "mean_harm": mean,
                    "ci95": sequence_bootstrap_interval(by_sequence),
                    "random_seed_range": [min(random_means), max(random_means)],
                    "excess_over_max_random": mean - max(random_means),
                    "gate_pass": passes,
                }
    # Compute per-frame ranks only when all four matched-budget categories exist.
    targets = defaultdict(dict)
    for row in complete:
        if row["category"] in CATEGORIES:
            targets[(row["video"], row["frame"], row["bank"])][row["category"]] = row
    correlations = defaultdict(list)
    for (video, frame, bank), values in targets.items():
        if set(values) != set(CATEGORIES):
            continue
        saliency, effects_per_token = [], []
        for category in CATEGORIES:
            row = values[category]
            masses = [
                layer[query][category]
                for layer in row["attention"]
                for query in ("foreground", "boundary")
                if layer[query] is not None
            ]
            if not masses:
                break
            saliency.append(float(np.mean(masses)))
            effects_per_token.append(row["harm_per_token"]["boundary_f"])
        if len(saliency) == 4:
            rho = rank_agreement(saliency, effects_per_token)["spearman"]
            if np.isfinite(rho):
                correlations[(bank, video)].append(rho)
    agreement = {}
    gate_disagreement = False
    for bank in ("conditioning", "recent"):
        values = {
            video: float(np.mean(rhos))
            for (name, video), rhos in correlations.items()
            if name == bank
        }
        if values:
            mean = float(np.mean(list(values.values())))
            passes = len(values) >= 8 and mean < 0.3
            gate_disagreement |= passes
            agreement[bank] = {
                "n_sequences": len(values),
                "rho": mean,
                "ci95": sequence_bootstrap_interval(values),
                "gate_pass": passes,
            }
    enough = set(sequences) == set(required_sequences)
    passed = enough and (gate_effect or gate_disagreement)
    return {
        "stage": "E1",
        "all_required_sequences": enough,
        "sequences": sequences,
        "completed_rows": len(complete),
        "skipped_rows": len(rows) - len(complete),
        "effects": effects,
        "attention_agreement": agreement,
        "gate_pass": passed,
        "decision": "proceed_to_E2" if passed else "stop_or_pivot",
        "full_study_complete": False,
        "hypotheses_confirmed": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    result = analyze(rows, json.loads(args.protocol.read_text())["pilot"]["sequences"])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
