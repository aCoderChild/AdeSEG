#!/usr/bin/env python3
"""Fit the temporal presence filter on build_fusion_data.py output and validate it across folds.

Per fold k, the fusion head and the filter are fitted on the other fold's clips
and both are scored on fold k by simulating the presence gate: a frame scores
``dice_open`` if the gate is open and 1 / 0 if it is closed on an empty /
polyp frame. No test video is used. The final filter is fitted on all clips.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MedSAM2")]

from modeling.detector_observation import PRESENCE_FEATURES, PresenceFilter
from scripts.train_presence_fusion import auroc, fit as fit_fusion


def sequences(data: pd.DataFrame, fusion) -> dict[str, torch.Tensor]:
    """Pad clips to [clips, frames] tensors; z is the per-frame presence log-odds."""
    features = torch.tensor(data[list(PRESENCE_FEATURES)].values, dtype=torch.float32)
    with torch.no_grad():
        fused = fusion(features)[:, 0]
    observed = torch.tensor(data.observed.values == 1)
    data = data.assign(z=torch.where(
        observed, features[:, 0] + features[:, 2], fused).numpy())
    clips = [group.sort_values("frame") for _, group in data.groupby(["fold", "clip"])]
    length = max(len(c) for c in clips)

    def pad(column, fill=0.0):
        out = np.full((len(clips), length), fill, dtype=np.float32)
        for i, c in enumerate(clips):
            out[i, : len(c)] = c[column].values
        return torch.tensor(out)

    return {
        "z": pad("z"), "observed": pad("observed") > 0, "gap": pad("gap"),
        "log_prior_variance": pad("log_prior_variance"), "present": pad("present"),
        "dice_open": pad("dice_open"), "valid": pad("present", -1) >= 0,
    }


def run_filter(model: PresenceFilter, seq: dict[str, torch.Tensor]) -> torch.Tensor:
    b, v = model.start(len(seq["z"]))
    beliefs = []
    for t in range(seq["z"].shape[1]):
        b, v = model(b, v, seq["z"][:, t], seq["observed"][:, t], seq["gap"][:, t], seq["log_prior_variance"][:, t])
        beliefs.append(b)
    return torch.stack(beliefs, dim=1)


def fit_filter(seq: dict[str, torch.Tensor], steps: int = 300) -> PresenceFilter:
    model = PresenceFilter()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    labels, valid = seq["present"][seq["valid"]], seq["valid"]
    weights = torch.where(labels == 1, 0.5 / labels.mean(), 0.5 / (1 - labels.mean()))
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(run_filter(model, seq)[valid], labels, reduction="none") * weights).mean()
        loss.backward()
        optimizer.step()
    return model


def gate_scores(score: torch.Tensor, seq: dict[str, torch.Tensor]) -> dict:
    valid = seq["valid"]
    gate, present, dice = score[valid] > 0, seq["present"][valid] == 1, seq["dice_open"][valid]
    frame_dice = torch.where(gate, dice, (~present).float())
    return {
        "dice": float(frame_dice.mean()),
        "present_dice": float(frame_dice[present].mean()),
        "absent_fp_rate": float(gate[~present].float().mean()),
        "present_detection_rate": float(gate[present].float().mean()),
        "auroc": auroc(score[valid], present.float()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop", nargs="*", default=[], choices=PRESENCE_FEATURES, help="Features set to zero (ablation).")
    args = parser.parse_args()
    torch.manual_seed(0)
    data = pd.read_csv(args.data)
    for name in args.drop:
        data[name] = 0.0

    def fusion_for(frame):
        rows = frame[frame.observed == 0]
        return fit_fusion(torch.tensor(rows[list(PRESENCE_FEATURES)].values, dtype=torch.float32),
                          torch.tensor(rows.present.values, dtype=torch.float32))

    report = {"dropped": args.drop, "clips": int(data.groupby(["fold", "clip"]).ngroups), "cross_fold": {}}
    for fold in sorted(data.fold.unique()):
        train, held = data[data.fold != fold], data[data.fold == fold]
        fusion = fusion_for(train)
        model = fit_filter(sequences(train, fusion))
        seq = sequences(held, fusion)
        with torch.no_grad():
            report["cross_fold"][int(fold)] = {
                "per_frame": gate_scores(seq["z"], seq),
                "filtered": gate_scores(run_filter(model, seq), seq),
            }
    fusion = fusion_for(data)
    model = fit_filter(sequences(data, fusion))
    report["parameters"] = {name: value.detach().tolist() for name, value in model.named_parameters()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "adseg_presence_filter_v1", "state_dict": model.state_dict()}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
