#!/usr/bin/env python3
"""
Failure analysis for video segmentation experiments.

Two entry-points:
  1. disconnect   — Find sequences where YOLO detection is good but mask IoU is poor
                    (detection-segmentation disconnect).  Reads per-sequence summary CSVs.

  2. timeline     — Plot Dice / IoU over frame index for one or more sequences, comparing
                    any number of methods side-by-side.

Usage examples
--------------
# Find disconnect sequences across all methods present in outputs/
python3 scripts/utils/failure_analysis.py disconnect

# Restrict to specific methods
python3 scripts/utils/failure_analysis.py disconnect \
    --methods MedSAM2_YOLO_BOX_MASK YOLO_SAM2_YOLO_BOX_FRAME

# Plot Dice timeline for seq10 and seq20 comparing two methods
python3 scripts/utils/failure_analysis.py timeline \
    --seqs 10 20 \
    --methods MedSAM2_YOLO_BOX_MASK YOLO_SAM2_YOLO_BOX_FRAME \
    --metric dice \
    --output_dir outputs/failure_analysis
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None

def _numeric_sort_key(name):
    parts = re.split(r"(\d+)", os.path.basename(str(name)))
    return tuple(int(p) if p.isdigit() else p for p in parts)


def _discover_methods(output_root):
    """Return sorted list of method names present in output_root."""
    if not os.path.isdir(output_root):
        return []
    return sorted(
        d for d in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, d))
    )


def _load_per_sequence_mask_iou(output_root, method_name, num_sequences=23):
    """Return {seq_num: mean_mask_iou} for a method, from eval/mask/seq*/metrics_avg.csv."""
    result = {}
    for seq_num in range(1, num_sequences + 1):
        f = os.path.join(
            output_root, method_name, "eval", "mask", f"seq{seq_num}", "metrics_avg.csv"
        )
        if not os.path.isfile(f):
            continue
        try:
            with open(f) as fh:
                reader = csv.DictReader(fh, skipinitialspace=True)
                for row in reader:
                    iou_val = row.get("iou", row.get("IoU", ""))
                    try:
                        result[seq_num] = float(iou_val)
                    except (ValueError, TypeError):
                        pass
                    break
        except Exception:
            pass
    return result


def _load_bbox_iou_per_sequence(output_root, method_name, num_sequences=23):
    """Return {seq_num: mean_bbox_iou} for a method, from eval/bbox/iou_per_sequence.csv."""
    result = {}
    per_seq_file = os.path.join(
        output_root, method_name, "eval", "bbox", "iou_per_sequence.csv"
    )
    if os.path.isfile(per_seq_file):
        try:
            with open(per_seq_file) as fh:
                reader = csv.DictReader(fh, skipinitialspace=True)
                for row in reader:
                    try:
                        seq_num = int(row["seq"])
                        result[seq_num] = float(row["mean_iou"])
                    except (KeyError, ValueError):
                        pass
        except Exception:
            pass
    return result


def _load_frame_metrics(output_root, method_name, seq_num, metric):
    """Return [(frame_idx, value), ...] from eval/mask/seq{N}/metrics.csv."""
    f = os.path.join(
        output_root, method_name, "eval", "mask", f"seq{seq_num}", "metrics.csv"
    )
    if not os.path.isfile(f):
        return []
    rows = []
    try:
        with open(f) as fh:
            reader = csv.DictReader(fh, skipinitialspace=True)
            for i, row in enumerate(reader):
                val_str = row.get(metric, "")
                try:
                    rows.append((i, float(val_str)))
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return rows


# ---------------------------------------------------------------------------
# Subcommand: disconnect
# ---------------------------------------------------------------------------

def cmd_disconnect(args):
    methods = args.methods or _discover_methods(args.output_root)
    if not methods:
        print(f"No methods found under {args.output_root}")
        return

    print(f"\nDetection-Segmentation Disconnect Analysis")
    print(f"Bbox IoU threshold (good detection): >= {args.bbox_thresh}")
    print(f"Mask IoU threshold (poor mask):      <  {args.mask_thresh}")
    print(f"Methods: {methods}\n")

    all_disconnect = {}

    for method in methods:
        bbox_iou = _load_bbox_iou_per_sequence(args.output_root, method, args.num_sequences)
        mask_iou = _load_per_sequence_mask_iou(args.output_root, method, args.num_sequences)

        disconnect = {}
        for seq_num in sorted(set(bbox_iou) & set(mask_iou)):
            b = bbox_iou[seq_num]
            m = mask_iou[seq_num]
            if b >= args.bbox_thresh and m < args.mask_thresh:
                gap = b - m
                disconnect[seq_num] = {"bbox_iou": b, "mask_iou": m, "gap": gap}

        if disconnect:
            print(f"  [{method}]  {len(disconnect)} disconnect sequence(s):")
            for seq_num, info in sorted(disconnect.items(), key=lambda x: -x[1]["gap"]):
                print(
                    f"    seq{seq_num:>2d}  bbox_iou={info['bbox_iou']:.3f}"
                    f"  mask_iou={info['mask_iou']:.3f}  gap={info['gap']:.3f}"
                )
        else:
            print(f"  [{method}]  no disconnect sequences found")

        all_disconnect[method] = disconnect

    # Save JSON summary
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "disconnect_summary.json")
        try:
            with open(out_path, "w") as fh:
                json.dump(all_disconnect, fh, indent=2)
            print(f"\nSummary written: {out_path}")
        except Exception as e:
            print(f"Could not write summary: {e}")


# ---------------------------------------------------------------------------
# Subcommand: timeline
# ---------------------------------------------------------------------------

def cmd_timeline(args):
    plt = _load_pyplot()
    if plt is None:
        print("matplotlib not available; cannot generate timeline plots.")
        sys.exit(1)

    methods = args.methods or _discover_methods(args.output_root)
    seqs = args.seqs or list(range(1, args.num_sequences + 1))
    metric = args.metric

    output_dir = args.output_dir or os.path.join(args.output_root, "failure_analysis", "timelines")
    os.makedirs(output_dir, exist_ok=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for seq_num in seqs:
        fig, ax = plt.subplots(figsize=(10, 4))
        has_data = False

        for i, method in enumerate(methods):
            rows = _load_frame_metrics(args.output_root, method, seq_num, metric)
            if not rows:
                continue
            has_data = True
            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            ax.plot(xs, ys, label=method, color=colors[i % len(colors)], linewidth=1.2)
            mean_val = np.mean(ys)
            ax.axhline(mean_val, color=colors[i % len(colors)], linestyle="--",
                       linewidth=0.7, alpha=0.6)

        if not has_data:
            plt.close(fig)
            print(f"  seq{seq_num}: no frame-level {metric} data found")
            continue

        ax.set_xlabel("Frame index")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"seq{seq_num} — {metric} over time")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"seq{seq_num}_{metric}_timeline.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nTimeline plots written to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Failure analysis for segmentation experiments."
    )
    parser.add_argument(
        "--output_root", default="outputs",
        help="Root containing method output folders (default: outputs)."
    )
    parser.add_argument(
        "--num_sequences", type=int, default=23,
        help="Total number of sequences (default: 23)."
    )

    sub = parser.add_subparsers(dest="command")

    # --- disconnect ---
    p_dis = sub.add_parser("disconnect", help="Find detection-segmentation disconnect sequences.")
    p_dis.add_argument("--methods", nargs="*", help="Method names to analyse (default: all).")
    p_dis.add_argument(
        "--bbox_thresh", type=float, default=0.7,
        help="Minimum bbox IoU to be considered a 'good detection' (default: 0.7)."
    )
    p_dis.add_argument(
        "--mask_thresh", type=float, default=0.4,
        help="Maximum mask IoU to be considered a 'poor mask' (default: 0.4)."
    )
    p_dis.add_argument(
        "--output_dir", default=None,
        help="Directory to write disconnect_summary.json (default: no file output)."
    )

    # --- timeline ---
    p_tl = sub.add_parser("timeline", help="Plot per-frame metric over time for given sequences.")
    p_tl.add_argument("--seqs", nargs="*", type=int, help="Sequence numbers (default: all).")
    p_tl.add_argument("--methods", nargs="*", help="Methods to plot (default: all).")
    p_tl.add_argument(
        "--metric", default="iou",
        help="Frame-level metric column to plot (default: iou)."
    )
    p_tl.add_argument(
        "--output_dir", default=None,
        help="Directory to save PNG plots (default: outputs/failure_analysis/timelines)."
    )

    args = parser.parse_args()

    if args.command == "disconnect":
        cmd_disconnect(args)
    elif args.command == "timeline":
        cmd_timeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
