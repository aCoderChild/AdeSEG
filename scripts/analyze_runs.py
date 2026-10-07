#!/usr/bin/env python3
"""Rebuild the PolypGen result tables from saved per-frame CSVs.

Each ``--run NAME=DIR`` points at an inference output with
``evaluation/metrics_per_frame.csv`` (evaluation/temporal.py) and the per-frame
diagnostics written by scripts/infer.py (``diagnostics/`` or ``masks/diagnostics/``).
Only propagated frames are scored. Per sequence: Dice; Dice on polyp frames; the
empty-frame false-positive rate; the lost fraction (polyp frames with Dice < 0.1);
reappearance Dice (polyp frames 1-10 after an absence); and high-motion Dice (polyp
frames whose image-feature change, taken from ``--motion_run``, exceeds
``--motion_cut``). Methods are the means over sequences; pooled (frame-level) Dice
and false-positive rates are printed separately.

Lost episodes are pooled over sequences: a run of consecutive polyp frames with
Dice < 0.1 (empty frames do not break it). An episode is recovered if a later
polyp frame of the sequence reaches Dice >= 0.5, and lost until the end if it
reaches the last polyp frame.

``--tau NAME=VALUE`` adds the output presence gate ``NAME+gate``: the mask is empty
wherever MedSAM2's object score is below VALUE. ``--compare A:B`` runs a paired
Wilcoxon test over sequences (SciPy defaults) with a sequence-bootstrap 95% interval
of the mean difference A - B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ("dice", "polyp", "fp", "lost", "reapp", "himot")


def load_run(directory: Path) -> pd.DataFrame:
    frames = pd.read_csv(directory / "evaluation" / "metrics_per_frame.csv")
    diagnostics_dir = next(d for d in (directory / "diagnostics", directory / "masks" / "diagnostics") if d.is_dir())
    diagnostics = pd.concat(
        pd.read_csv(path).assign(sequence=path.stem) for path in sorted(diagnostics_dir.glob("*.csv"))
    )
    columns = ["sequence", "frame", "status", "object_score", "feature_change_mean"]
    frames = frames.merge(diagnostics.reindex(columns=columns), on=["sequence", "frame"], how="left")
    return frames[frames["status"] == "propagated"].reset_index(drop=True)


def gate(frames: pd.DataFrame, tau: float) -> pd.DataFrame:
    frames = frames.copy()
    closed = frames["object_score"] < tau
    frames.loc[closed, "dice"] = np.where(frames.loc[closed, "gt_present"], 0.0, 1.0)
    frames.loc[closed, "predicted_present"] = False
    return frames


def per_sequence(frames: pd.DataFrame, motion: pd.Series, motion_cut: float) -> pd.DataFrame:
    polyp = frames[frames["gt_present"]]
    since = polyp["frames_since_reappearance"]
    moving = polyp[motion.reindex(polyp.index) > motion_cut]
    mean = lambda rows, values: values.groupby(rows["sequence"]).mean()
    return pd.DataFrame({
        "dice": mean(frames, frames["dice"]),
        "polyp": mean(polyp, polyp["dice"]),
        "fp": mean(frames[~frames["gt_present"]], frames.loc[~frames["gt_present"], "predicted_present"].astype(float)),
        "lost": mean(polyp, (polyp["dice"] < 0.1).astype(float)),
        "reapp": mean(polyp[(since >= 1) & (since <= 10)], polyp.loc[(since >= 1) & (since <= 10), "dice"]),
        "himot": mean(moving, moving["dice"]),
    })


def episodes(frames: pd.DataFrame) -> dict[str, float]:
    polyp = frames[frames["gt_present"]]
    found = []
    for _, sequence in polyp.groupby("sequence", sort=False):
        dice, length = sequence["dice"].to_numpy(), 0
        for index, value in enumerate(dice):
            if value < 0.1:
                length += 1
                continue
            if length:
                found.append((length, bool((dice[index:] >= 0.5).any()), False))
                length = 0
        if length:
            found.append((length, False, True))
    lengths, recovered, until_end = (np.array(column, dtype=float) for column in zip(*found)) if found else ([],) * 3
    return {
        "lost_frames": float((polyp["dice"] < 0.1).mean()),
        "episodes": len(found),
        "mean_episode": float(np.mean(lengths)) if found else float("nan"),
        "recovered": float(np.mean(recovered)) if found else float("nan"),
        "lost_until_end": float(np.mean(until_end)) if found else float("nan"),
    }


def paired(first: pd.DataFrame, second: pd.DataFrame, samples: int, rng) -> dict[str, dict]:
    result = {}
    for metric in METRICS:
        both = pd.concat([first[metric], second[metric]], axis=1, keys=["a", "b"]).dropna()
        difference = (both["a"] - both["b"]).to_numpy()
        boot = rng.choice(difference, (samples, difference.size)).mean(axis=1) if difference.size else np.array([np.nan])
        result[metric] = {
            "diff": float(difference.mean()) if difference.size else float("nan"),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "higher": int((difference > 0).sum()), "lower": int((difference < 0).sum()), "n": int(difference.size),
            "p": float(wilcoxon(difference).pvalue) if np.any(difference != 0) else 1.0,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--motion_run", default=None, help="run whose feature_change_mean defines motion (default: first run that logs it)")
    parser.add_argument("--motion_cut", type=float, default=0.117)
    parser.add_argument("--tau", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--compare", nargs="*", default=[], metavar="A:B")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        name, directory = item.split("=", 1)
        frames = load_run(Path(directory))
        runs[name] = frames[frames["sequence"].isin(args.sequences)] if args.sequences else frames
    for item in args.tau:
        name, value = item.split("=", 1)
        runs[f"{name}+gate"] = gate(runs[name], float(value))
    motion_name = args.motion_run or next(
        (name for name, frames in runs.items() if frames["feature_change_mean"].notna().any()), None)
    motion_key = ["sequence", "frame"]
    motion_table = runs[motion_name][motion_key + ["feature_change_mean"]] if motion_name else None

    tables, summary = {}, {"motion_run": motion_name, "motion_cut": args.motion_cut, "methods": {}, "paired": {}}
    for name, frames in runs.items():
        motion = (frames[motion_key].merge(motion_table, on=motion_key, how="left")["feature_change_mean"]
                  if motion_table is not None else pd.Series(np.nan, index=frames.index))
        motion.index = frames.index
        tables[name] = per_sequence(frames, motion, args.motion_cut)
        summary["methods"][name] = {
            "sequences": int(tables[name]["dice"].notna().sum()),
            **{metric: float(tables[name][metric].mean()) for metric in METRICS},
            **episodes(frames),
            "pooled_dice": float(frames["dice"].mean()),
            "pooled_polyp": float(frames.loc[frames["gt_present"], "dice"].mean()),
            "pooled_fp": float(frames.loc[~frames["gt_present"], "predicted_present"].mean()),
        }
    rng = np.random.default_rng(args.seed)
    for pair in args.compare:
        first, second = pair.split(":")
        summary["paired"][pair] = paired(tables[first], tables[second], args.bootstrap, rng)

    print("| Method | Dice | Polyp-frame Dice | Empty-frame FP | Lost | Reappearance | High motion "
          "| Lost (pooled) | Mean episode | Recovered | Lost until end |")
    print("|---" * 11 + "|")
    for name, row in summary["methods"].items():
        values = [row[m] for m in METRICS] + [row["lost_frames"], row["mean_episode"], row["recovered"], row["lost_until_end"]]
        print(f"| {name} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    print("\n| Method | Pooled Dice | Pooled polyp-frame Dice | Pooled empty-frame FP |\n|---|---|---|---|")
    for name, row in summary["methods"].items():
        print(f"| {name} | {row['pooled_dice']:.3f} | {row['pooled_polyp']:.3f} | {row['pooled_fp']:.3f} |")
    for pair, metrics in summary["paired"].items():
        print(f"\n{pair} (A - B, higher/lower of n sequences, Wilcoxon p, bootstrap 95% CI)")
        for metric, s in metrics.items():
            print(f"  {metric}: {s['diff']:+.4f} ({s['higher']}/{s['lower']} of {s['n']}), p = {s['p']:.3f}, "
                  f"CI [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(tables, names=["method", "sequence"]).to_csv(args.output_dir / "per_sequence.csv")
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
