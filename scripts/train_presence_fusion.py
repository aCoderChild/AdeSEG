#!/usr/bin/env python3
"""Fit the presence-fusion head on build_fusion_data.py output (class-balanced logistic BCE)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MedSAM2")]

from modeling.detector_observation import PRESENCE_FEATURES, PresenceFusion


def fit(features, labels):
    model = PresenceFusion()
    weights = torch.where(labels == 1, 0.5 / labels.mean(), 0.5 / (1 - labels.mean()))
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(model(features)[:, 0], labels, reduction="none") * weights).mean()
        loss = loss + 1e-3 * model.linear.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return model


def auroc(scores, labels):
    order = scores.argsort()
    ranks = torch.empty_like(scores); ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    positives = labels.sum()
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop", nargs="*", default=[], choices=PRESENCE_FEATURES, help="Features set to zero (ablation).")
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.data)))
    features = torch.tensor([[float(r[name]) for name in PRESENCE_FEATURES] for r in rows])
    for name in args.drop:
        features[:, PRESENCE_FEATURES.index(name)] = 0.0
    labels = torch.tensor([float(r["present"]) for r in rows])
    folds = torch.tensor([int(r["fold"]) for r in rows])

    report = {"dropped": args.drop, "frames": len(rows), "present_fraction": float(labels.mean()), "cross_fold": {}}
    for fold in folds.unique().tolist():
        held = folds == fold
        model = fit(features[~held], labels[~held])
        with torch.no_grad():
            report["cross_fold"][fold] = {
                "fused_auroc": auroc(model(features[held])[:, 0], labels[held]),
                "object_score_auroc": auroc(features[held, 0], labels[held]),
            }
    model = fit(features, labels)
    report["weights"] = dict(zip(PRESENCE_FEATURES, model.linear.weight[0].tolist()))
    report["bias"] = model.linear.bias.item()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "adseg_presence_fusion_v1", "features": PRESENCE_FEATURES, "state_dict": model.state_dict()}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
