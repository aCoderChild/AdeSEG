#!/usr/bin/env python3
"""Calibrate the Kalman reliability rho = sigmoid(a * object score + c) on real dev videos.

Input: an ``infer.py`` Kalman run (no detector, gate off) whose ``masks/diagnostics``
hold MedSAM2's object score per frame, and its ``evaluation/metrics_per_frame.csv``.
Frames: every propagated frame of the ``protocol_a.json`` split whose presence gate is
open (the frames the Kalman memory writes). A frame is reliable if it contains a polyp
and its IoU is >= 0.5; wrong-object, partial and false-positive frames are not.
Plain (unweighted) logistic regression, so rho is a probability. Quality is reported
leave-one-sequence-out: each sequence is scored by a fit on the others.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_frames(run: Path, split: str) -> pd.DataFrame:
    sequences = {f"seq{n}" for n in json.loads((ROOT / "datasets/splits/polypgen/protocol_a.json").read_text())[split]}
    metrics = pd.read_csv(run / "evaluation/metrics_per_frame.csv")
    metrics = metrics[metrics.sequence.isin(sequences) & (metrics.frames_after_prompt >= 1)]
    diagnostics = pd.concat([
        pd.read_csv(path).assign(sequence=path.stem)
        for path in sorted((run / "masks/diagnostics").glob("seq*.csv")) if path.stem in sequences
    ])
    frames = metrics.merge(diagnostics[["sequence", "frame", "object_score"]], on=["sequence", "frame"])
    frames = frames[frames.object_score > 0].copy()  # presence gate open
    frames["reliable"] = (frames.gt_present & (frames.iou >= 0.5)).astype(float)
    return frames


def fit(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    x, y = torch.tensor(scores, dtype=torch.float64), torch.tensor(labels, dtype=torch.float64)
    params = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([params], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(params[0] * x + params[1], y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(params[0].detach()), float(params[1].detach())


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    ranks = pd.Series(scores).rank().to_numpy()
    positives = labels.sum()
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from_run", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = load_frames(args.from_run, args.split)
    held_out = np.zeros(len(frames))
    for sequence in frames.sequence.unique():
        test = (frames.sequence == sequence).to_numpy()
        slope, offset = fit(frames.object_score[~test].to_numpy(), frames.reliable[~test].to_numpy())
        held_out[test] = 1 / (1 + np.exp(-(slope * frames.object_score[test].to_numpy() + offset)))
    labels = frames.reliable.to_numpy()
    slope, offset = fit(frames.object_score.to_numpy(), labels)
    report = {
        "slope": slope,
        "offset": offset,
        "object_score_at_rho_0.5": -offset / slope,
        "split": args.split,
        "source": str(args.from_run),
        "frames": len(frames),
        "reliable_fraction": float(labels.mean()),
        "loso_auroc": auroc(held_out, labels),
        "loso_brier": float(np.mean((held_out - labels) ** 2)),
        "brier_of_constant": float(labels.mean() * (1 - labels.mean())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
