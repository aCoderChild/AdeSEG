#!/usr/bin/env python3
"""Fit (alpha, beta) for the soft score-scaled Kalman gain on dev diagnostics.

Model: scale = sigmoid(alpha * object_score + beta) predicts whether a frame is
"correctly tracked" (Dice >= tau on polyp frames; correctly empty otherwise).
Fit by logistic regression on dev sequences only. Prints a one-line command
that can be fed to scripts/infer.py as ``--score_scale alpha beta``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true); fp = np.cumsum(1 - y_true)
    tpr = tp / max(y_true.sum(), 1)
    fpr = fp / max((1 - y_true).sum(), 1)
    return float(np.trapezoid(tpr, fpr))


DEV_SEQUENCES = {f"seq{i}" for i in range(1, 16)}


def load_diag_scores(diag_dir: Path):
    out = {}
    for path in sorted(diag_dir.glob("seq*.csv")):
        sequence = path.stem
        if sequence not in DEV_SEQUENCES:
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                if row["status"] != "propagated" or not row["object_score"]:
                    continue
                out[(sequence, row["frame"])] = float(row["object_score"])
    return out


def load_dice(csv_path: Path):
    out = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row["sequence"] not in DEV_SEQUENCES:
                continue
            if row["frames_after_prompt"] in ("", "nan"):
                continue
            out[(row["sequence"], row["frame"])] = {
                "dice": float(row["dice"]),
                "gt_present": row["gt_present"].lower() == "true",
            }
    return out


def label(dice: float, gt_present: bool, tau: float) -> int:
    if gt_present:
        return int(dice >= tau)
    return int(dice >= 0.5)  # Dice is 1 on correctly empty frames


def fit(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    def neg_log_likelihood(params):
        alpha, beta = params
        logits = alpha * scores + beta
        return float(np.logaddexp(0, -logits * (2 * labels - 1)).mean())

    result = minimize(neg_log_likelihood, x0=[0.5, -2.0], method="L-BFGS-B")
    alpha, beta = result.x
    probs = 1 / (1 + np.exp(-(alpha * scores + beta)))
    auroc = roc_auc_score(labels, probs) if labels.min() != labels.max() else float("nan")
    return float(alpha), float(beta), float(auroc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag_dir", type=Path, default=Path.home() / "Library/Caches/adseg_work/16_dev_kalman_motion_logged/masks/diagnostics")
    parser.add_argument("--dice_csv", type=Path, default=Path.home() / "Library/Caches/adseg_work/analysis/10_kalman_memory.csv")
    parser.add_argument("--tau", type=float, default=0.5, help="Dice threshold for 'correctly tracked'.")
    parser.add_argument("--out", type=Path, default=Path.home() / "Library/Caches/adseg_work/analysis/soft_gain.json")
    args = parser.parse_args()

    scores = load_diag_scores(args.diag_dir)
    dice = load_dice(args.dice_csv)
    shared = scores.keys() & dice.keys()
    if not shared:
        raise SystemExit("No overlap between diagnostics and Dice CSV.")

    xs, ys = [], []
    present_xs, present_ys = [], []
    for key in shared:
        d = dice[key]
        ys.append(label(d["dice"], d["gt_present"], args.tau))
        xs.append(scores[key])
        if d["gt_present"]:
            present_xs.append(scores[key])
            present_ys.append(int(d["dice"] >= args.tau))
    xs = np.asarray(xs); ys = np.asarray(ys)
    present_xs = np.asarray(present_xs); present_ys = np.asarray(present_ys)

    alpha_all, beta_all, auroc_all = fit(xs, ys)
    alpha_present, beta_present, auroc_present = fit(present_xs, present_ys)

    out = {
        "n_frames": int(len(xs)),
        "tau": args.tau,
        "all_frames": {"alpha": alpha_all, "beta": beta_all, "auroc": auroc_all,
                       "positive_rate": float(ys.mean())},
        "present_only": {"alpha": alpha_present, "beta": beta_present, "auroc": auroc_present,
                         "positive_rate": float(present_ys.mean()), "n": int(len(present_xs))},
        "example_run": (f"python3 scripts/infer.py --score_scale {alpha_all:.4f} {beta_all:.4f} ..."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
