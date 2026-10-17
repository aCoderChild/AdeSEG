#!/usr/bin/env python3
"""Train an exploratory MLP utility predictor from cached leave-one-out records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.utility_retrieval import paired_wilcoxon, rank_agreement, sequence_bootstrap_interval
from modeling.utility_memory import UtilityController, UtilityHead, save_utility_head
from training.tracking import finish, log, start_wandb


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--regression-weight", type=float, default=0.1)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--wandb-project", default="ade-seg")
    parser.add_argument("--wandb-entity", default="adenoid-hypertrophy")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-base-url", default="https://api.wandb.ai")
    return parser.parse_args()


def load_records(path):
    records = torch.load(path, map_location="cpu", weights_only=False)
    if not records:
        raise ValueError(f"No utility records in {path}")
    return records


def group_frames(records):
    frames = defaultdict(list)
    for record in records:
        frames[(record["video"], int(record["frame"]))].append(record)
    return frames


def normalized(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def frame_loss(controller, records, regression_weight):
    features = torch.stack([record["features"] for record in records]).to(next(controller.parameters()).device)
    utility = normalized([record["utility"] for record in records]).to(features.device)
    prediction = controller.bias_scale * controller.head(features)
    differences = utility[:, None] - utility[None, :]
    upper = torch.triu(torch.ones_like(differences, dtype=torch.bool), diagonal=1) & (differences != 0)
    if upper.any():
        pred_diff = prediction[:, None] - prediction[None, :]
        ranking = F.softplus(-differences.sign()[upper] * pred_diff[upper]).mean()
    else:
        ranking = prediction.sum() * 0
    regression = F.smooth_l1_loss(prediction, utility)
    return ranking + regression_weight * regression, ranking, regression


def auroc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    positive, negative = labels.sum(), (~labels).sum()
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


@torch.no_grad()
def evaluate(controller, records):
    by_frame = group_frames(records)
    by_sequence = defaultdict(lambda: defaultdict(list))
    harmful_labels, harmful_scores = [], []
    for (video, _), frame_records in by_frame.items():
        features = torch.stack([item["features"] for item in frame_records]).to(next(controller.parameters()).device)
        prediction = (controller.bias_scale * controller.head(features)).cpu().numpy()
        utility = np.array([item["utility"] for item in frame_records])
        criteria = {
            "probe": prediction,
            "attention": np.array([item["attention"] for item in frame_records]),
            "similarity": np.array([item["semantic_similarity"] for item in frame_records]),
            "confidence": np.array([
                item["confidence"] if item["confidence"] is not None else float("nan")
                for item in frame_records
            ]),
        }
        for name, values in criteria.items():
            by_sequence[video][name].append(rank_agreement(values, utility)["spearman"])
        harmful_labels.extend(utility < 0)
        harmful_scores.extend(-prediction)  # higher score means more likely harmful
    sequence_scores = {
        name: {
            video: float(np.nanmean(values[name]))
            for video, values in by_sequence.items()
        }
        for name in ("probe", "attention", "similarity", "confidence")
    }
    return sequence_scores, auroc(harmful_labels, harmful_scores)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    train_records = load_records(args.train_records)
    evaluation_records = load_records(args.evaluation_records)
    feature_dim = int(train_records[0]["features"].numel())
    if feature_dim != 2 * 256 + 64 + 4:
        raise ValueError(f"Unexpected utility feature width: {feature_dim}")
    controller = UtilityController(UtilityHead()).to(args.device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.learning_rate)
    frames = list(group_frames(train_records).values())
    setup = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    setup.update({"train_frames": len(frames), "train_units": len(train_records),
                  "evaluation_units": len(evaluation_records)})
    (args.output_dir / "setup.json").write_text(json.dumps(setup, indent=2) + "\n")
    run = start_wandb(args.wandb_project, args.wandb_entity, args.wandb_name,
                      args.wandb_mode, args.wandb_base_url, setup, args.output_dir,
                      ("utility", "E5", f"seed-{args.seed}"))
    try:
        for epoch in range(args.epochs):
            order = torch.randperm(len(frames)).tolist()
            losses = []
            for index in order:
                optimizer.zero_grad(set_to_none=True)
                loss, ranking, regression = frame_loss(controller, frames[index], args.regression_weight)
                loss.backward()
                optimizer.step()
                losses.append((float(loss.detach()), float(ranking.detach()), float(regression.detach())))
            means = np.mean(losses, axis=0)
            log(run, "train", {"loss": means[0], "ranking": means[1], "regression": means[2]}, epoch)

        sequence_scores, harmful_auroc = evaluate(controller, evaluation_records)
        best_baseline = max(
            ("attention", "similarity", "confidence"),
            key=lambda name: np.nanmean(list(sequence_scores[name].values())),
        )
        comparison = paired_wilcoxon(sequence_scores["probe"], sequence_scores[best_baseline])
        differences = {
            name: sequence_scores["probe"][name] - sequence_scores[best_baseline][name]
            for name in sequence_scores["probe"]
            if name in sequence_scores[best_baseline]
        }
        summary = {
            "seed": args.seed,
            "sequence_mean_spearman": {
                name: float(np.nanmean(list(values.values()))) for name, values in sequence_scores.items()
            },
            "sequence_bootstrap_95ci": {
                name: sequence_bootstrap_interval(values) for name, values in sequence_scores.items()
            },
            "harmful_unit_auroc": harmful_auroc,
            "learned_bias_scale": float(controller.bias_scale.detach().cpu()),
            "best_baseline": best_baseline,
            "probe_vs_best_baseline": comparison,
            "median_sequence_improvement": float(np.nanmedian(list(differences.values()))),
            # Probe correlation is not E5 causal segmentation evidence, nor the
            # E4 linear-probe result with preregistered controls.
            "E4_exploratory_mlp_beats_recorded_proxies": bool(
                comparison["p"] < 0.05 and np.nanmedian(list(differences.values())) > 0
            ),
        }
        save_utility_head(controller, args.output_dir / "utility_head.pt", summary)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        log(run, "E5", {**summary["sequence_mean_spearman"], "harmful_auroc": harmful_auroc,
                         "p_vs_best": comparison["p"]}, args.epochs)
    finally:
        finish(run)


if __name__ == "__main__":
    main()
