"""Paired experiment analysis: input-protocol matching and sequence-level deltas."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .protocol import load_protocol, comparison_key

def _scores(path, metric):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"sequence", metric}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires sequence and selected metric columns")
        result = {}
        for row in reader:
            name = row["sequence"]
            if not name or name in result:
                raise ValueError("Sequence IDs must be nonempty and unique")
            result[name] = float(row[metric])
    if not result or not all(np.isfinite(x) for x in result.values()):
        raise ValueError("No scores or nonfinite scores")
    return result

def paired_comparison(baseline_protocol, candidate_protocol, baseline_csv, candidate_csv,
                      *, metric="dice", n_bootstrap=10000, seed=0):
    _, base = load_protocol(baseline_protocol)
    _, candidate = load_protocol(candidate_protocol)
    if comparison_key(base) != comparison_key(candidate):
        raise ValueError("Protocol mismatch: evaluation, dataset, backbone checkpoint, split or prompt conditions differ")
    if base["track"] != candidate["track"]:
        raise ValueError("Ablations require the same backbone/research track")
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    left, right = _scores(baseline_csv, metric), _scores(candidate_csv, metric)
    if left.keys() != right.keys():
        raise ValueError("Baseline and candidate evaluated different sequences")
    names = sorted(left)
    differences = np.asarray([right[name]-left[name] for name in names])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(names), size=(n_bootstrap, len(names)))
    means = differences[indices].mean(axis=1)
    return {
        "metric": metric, "sequences": len(names),
        "baseline_mean": float(np.mean([left[x] for x in names])),
        "candidate_mean": float(np.mean([right[x] for x in names])),
        "paired_mean_difference": float(differences.mean()),
        "bootstrap_95_ci": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "sequence_differences": dict(zip(names, differences.tolist())),
        "bootstrap_seed": seed, "bootstrap_samples": n_bootstrap,
        "warning": "Exploratory paired bootstrap; sequence independence must be established by dataset provenance",
    }

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-protocol", type=Path, required=True)
    p.add_argument("--candidate-protocol", type=Path, required=True)
    p.add_argument("--baseline-csv", type=Path, required=True)
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--metric", default="dice")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    result = paired_comparison(
        args.baseline_protocol, args.candidate_protocol,
        args.baseline_csv, args.candidate_csv,
        metric=args.metric, n_bootstrap=args.bootstrap, seed=args.seed)
    if args.output.exists():
        p.error("Refusing to overwrite existing ablation report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
