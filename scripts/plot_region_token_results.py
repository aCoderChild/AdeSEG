#!/usr/bin/env python3
"""Create reproducible figures for Run155 and the token-key validation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


COLORS = {
    "native": "#596579",
    "generic": "#7b8db5",
    "region": "#168a82",
    "token": "#d66a4a",
    "global": "#d49b31",
    "positive": "#168a82",
    "negative": "#c65b55",
    "ink": "#253047",
    "muted": "#637087",
    "grid": "#d9dee7",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def finish_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run155_figures(summary: dict, output_dir: Path) -> None:
    per_sequence = summary["per_sequence"]
    seqs = list(summary["sequences"])
    deltas_pp = {
        seq: 100.0 * (
            per_sequence[seq]["region_constrained_tome"]["present_dice"]
            - per_sequence[seq]["native"]["present_dice"]
        )
        for seq in seqs
    }
    contrasts = summary["paired_contrasts"]

    fig, (ax_ci, ax_seq) = plt.subplots(
        1, 2, figsize=(13.6, 8.0), gridspec_kw={"width_ratios": [1.05, 1.35]}
    )
    fig.suptitle("Run155 · region-constrained token merging", fontsize=18, weight="bold", color=COLORS["ink"])
    fig.text(
        0.5, 0.925,
        "21 PolypGen sequences · sequence is the statistical unit · exploratory",
        ha="center", fontsize=10.5, color=COLORS["muted"],
    )

    contrast_keys = ["regional_minus_native", "regional_minus_generic"]
    labels = [
        f"Regional − native ({contrasts[contrast_keys[0]]['positive_sequence_wins']}/"
        f"{contrasts[contrast_keys[0]]['n_sequences']} wins; "
        f"p={contrasts[contrast_keys[0]]['holm_adjusted_p']:.3f})",
        f"Regional − generic ({contrasts[contrast_keys[1]]['positive_sequence_wins']}/"
        f"{contrasts[contrast_keys[1]]['n_sequences']} wins; "
        f"p={contrasts[contrast_keys[1]]['holm_adjusted_p']:.3f})",
    ]
    y_positions = [1, 0]
    for y, label, key in zip(y_positions, labels, contrast_keys, strict=True):
        item = contrasts[key]
        mean = 100 * item["mean_difference"]
        lo, hi = [100 * x for x in item["sequence_bootstrap_95ci"]]
        ax_ci.errorbar(
            mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o", markersize=8,
            capsize=5, lw=2.2, color=COLORS["region" if "native" in key else "generic"],
        )
        ax_ci.annotate(f"{mean:+.2f} pp", (mean, y), xytext=(0, 11),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=8.5, color=COLORS["ink"])
    ax_ci.axvline(0, color=COLORS["ink"], lw=1.1, alpha=0.8)
    ax_ci.set_yticks(y_positions, labels)
    ax_ci.set_xlim(-1.8, 9.3)
    ax_ci.set_ylim(-0.8, 1.8)
    ax_ci.set_xlabel("Paired mean Dice difference (percentage points)")
    ax_ci.set_title("Mean effects and 95% sequence-bootstrap CIs", loc="left", weight="bold")
    ax_ci.grid(axis="x", color=COLORS["grid"], lw=0.8)
    ax_ci.set_axisbelow(True)

    other_seqs = [seq for seq in seqs if seq != "seq8"]
    vals = [deltas_pp[seq] for seq in other_seqs]
    ys = np.arange(len(other_seqs))
    mean_without_seq8 = float(np.mean(vals))
    bar_colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in vals]
    ax_seq.barh(ys, vals, height=0.66, color=bar_colors, alpha=0.88)
    ax_seq.axvline(0, color=COLORS["ink"], lw=1.1, alpha=0.8)
    ax_seq.axvline(mean_without_seq8, color=COLORS["region"], lw=1.6, ls="--")
    ax_seq.set_yticks(ys, other_seqs)
    ax_seq.invert_yaxis()
    ax_seq.set_xlim(-2.2, 4.1)
    ax_seq.set_xlabel("Regional − native Dice difference (percentage points; seq8 omitted)")
    ax_seq.set_title("Per-sequence deltas without the outlier", loc="left", weight="bold")
    ax_seq.grid(axis="x", color=COLORS["grid"], lw=0.8)
    ax_seq.set_axisbelow(True)

    fig.text(
        0.5, 0.035,
        f"seq8 delta (included in left panel): {deltas_pp['seq8']:+.2f} pp.  "
        f"Mean delta across the other 20 sequences: {mean_without_seq8:+.2f} pp.",
        ha="center", va="bottom", fontsize=9, color=COLORS["ink"],
    )
    fig.subplots_adjust(top=0.84, bottom=0.20, wspace=0.42)
    finish_figure(fig, output_dir, "run155_effects")


def current_validation_figure(results: dict, output_dir: Path) -> None:
    native = results["conditions"]["native"]
    token = results["conditions"]["token_conditioned"]
    global_gate = results["conditions"]["global_gate"]
    seqs = ["seq13", "seq14", "seq15"]
    token_d = [1000 * (token[s]["present_dice"] - native[s]["present_dice"]) for s in seqs]
    global_d = [1000 * (global_gate[s]["present_dice"] - native[s]["present_dice"]) for s in seqs]
    token_contrast = results["contrasts"]["token_conditioned_minus_native"]
    global_contrast = results["contrasts"]["global_gate_minus_native"]

    fig, ax = plt.subplots(figsize=(11.8, 5.7))
    fig.suptitle("Current run · token-key adapter validation", fontsize=18, weight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.91, "Only validation seq13–15 completed; test seq16–23 was withheld by the frozen screen.",
             ha="center", fontsize=10.5, color=COLORS["muted"])
    y = np.arange(3)
    ax.axvline(0, color=COLORS["ink"], lw=1.2)
    ax.scatter(token_d, y + 0.10, s=80, marker="o", color=COLORS["token"], label="Token-conditioned adapter")
    ax.scatter(global_d, y - 0.10, s=80, marker="s", color=COLORS["global"], label="Global-gate control")
    for i in range(3):
        ax.plot([token_d[i], global_d[i]], [y[i] + 0.10, y[i] - 0.10], color=COLORS["grid"], lw=1.2, zorder=0)
    ax.set_yticks(y, seqs)
    ax.invert_yaxis()
    ax.set_xlim(-0.78, 0.27)
    ax.set_xlabel("Present Dice change vs native SAM2 (milliDice; 1 unit = 0.001 Dice)")
    ax.set_title("Paired sequence-level differences; zero means tied with native", loc="left", weight="bold")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    token_ci = [1000 * x for x in token_contrast["bootstrap_95_ci"]]
    global_ci = [1000 * x for x in global_contrast["bootstrap_95_ci"]]
    token_mean = 1000 * token_contrast["mean"]
    global_mean = 1000 * global_contrast["mean"]
    note = (
        f"Mean token − native: {token_mean:+.3f} mDice; 95% CI [{token_ci[0]:+.3f}, {token_ci[1]:+.3f}], "
        f"Holm p={results['holm_adjusted_p']['token_conditioned_minus_native']:.2f}; wins 1/3.\n"
        f"Mean global − native: {global_mean:+.3f} mDice; 95% CI [{global_ci[0]:+.3f}, {global_ci[1]:+.3f}].\n"
        "Both intervals include zero; false alarms were identical at 74/93 frames in all arms."
    )
    fig.text(0.13, 0.02, note, ha="left", va="bottom", fontsize=9.4, color=COLORS["ink"])
    fig.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.24)
    finish_figure(fig, output_dir, "current_validation_deltas")


def method_diagram(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.2, 6.6))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    fig.suptitle("Two token-level strategies explored", fontsize=19, weight="bold", color=COLORS["ink"], y=0.97)

    def box(x, y, w, h, text, color, fontsize=10.3):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                               linewidth=1.5, edgecolor=color, facecolor=color + "18")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                color=COLORS["ink"], wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.4, color=COLORS["muted"]))

    ax.text(0.35, 5.9, "RUN155 · REGION-CONSTRAINED ToMe", fontsize=12, weight="bold", color=COLORS["region"])
    box(0.35, 4.25, 2.7, 1.15, "Stored spatial memory\n+ predicted-mask roles", COLORS["region"])
    box(3.65, 4.25, 2.8, 1.15, "Local-window matching\nwithin predicted role", COLORS["region"])
    box(7.05, 4.25, 2.7, 1.15, "Merge similar K/V tokens\n(pointer tokens retained)", COLORS["region"])
    box(10.35, 4.25, 4.1, 1.15, "SAM2 memory attention reads\na compressed spatial bank", COLORS["region"])
    arrow(3.05, 4.82, 3.65, 4.82)
    arrow(6.45, 4.82, 7.05, 4.82)
    arrow(9.75, 4.82, 10.35, 4.82)
    ax.text(0.45, 3.77, "21-sequence result: +2.45 percentage points vs native; interval includes zero; seq8 drives much of the mean.",
            fontsize=9.4, color=COLORS["ink"])

    ax.plot([0.3, 14.8], [3.35, 3.35], color=COLORS["grid"], lw=1.2)
    ax.text(0.35, 2.85, "PREVIOUS STUDY · TOKEN-CONDITIONED KEY RESIDUAL", fontsize=12, weight="bold", color=COLORS["token"])
    box(0.35, 1.15, 2.7, 1.15, "Each memory key x\n+ object pointer context p", COLORS["token"])
    box(3.65, 1.15, 2.8, 1.15, "Rank-16 residual with\ncontinuous per-token gates", COLORS["token"])
    box(7.05, 1.15, 2.7, 1.15, "Frozen K projection\nand native attention", COLORS["token"])
    box(10.35, 1.15, 4.1, 1.15, "All tokens / frames retained;\nSAM2 weights remain frozen", COLORS["token"])
    arrow(3.05, 1.72, 3.65, 1.72)
    arrow(6.45, 1.72, 7.05, 1.72)
    arrow(9.75, 1.72, 10.35, 1.72)
    ax.text(0.45, 0.62, "Validation seq13–15: −0.000169 Dice vs native; frozen release screen failed; test was withheld.",
            fontsize=9.4, color=COLORS["ink"])
    ax.text(15.0, 0.13, "Run155 source hash recovered; efficacy remains exploratory and non-significant.",
            fontsize=7.8, color=COLORS["muted"], ha="right")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.89, bottom=0.06)
    finish_figure(fig, output_dir, "method_comparison_diagram")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run155-summary", type=Path, required=True)
    parser.add_argument("--current-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run155_figures(load_json(args.run155_summary), args.output_dir)
    current_validation_figure(load_json(args.current_validation), args.output_dir)
    method_diagram(args.output_dir)
    print(f"Wrote three figure pairs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
