#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def normalize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for col in columns:
        if col not in normalized.columns:
            continue
        values = pd.to_numeric(normalized[col], errors="coerce")
        min_value = values.min()
        max_value = values.max()
        if pd.isna(min_value) or pd.isna(max_value):
            normalized[col] = 0.0
        elif max_value == min_value:
            normalized[col] = 0.0
        else:
            normalized[col] = (values - min_value) / (max_value - min_value)
    return normalized


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_summary(summary_csv: Path, out_dir: Path, normalize: bool = True) -> None:
    df = read_csv_clean(summary_csv)
    if df.empty:
        return

    # Use these metrics (remove centroid_disp_mean) and plot metrics on x-axis,
    # with one bar per pipeline/method for each metric.
    metrics = ["mean_dice", "mean_iou", "mean_ciou", "pred_area_std_norm"]
    df_plot = df.copy()
    if normalize:
        df_plot = normalize_columns(df_plot, metrics)

    # create a metrics x pipelines table: rows=metrics, cols=pipeline
    pivot = df_plot.set_index("pipeline")[metrics].T

    # ensure a sane pipeline order if present
    desired_order = ["MedSAM2", "MedSAM2_first_frame", "YOLO_SAM2"]
    cols = [c for c in desired_order if c in pivot.columns] + [c for c in pivot.columns if c not in desired_order]
    pivot = pivot[cols]

    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_title("Overall Summary Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.legend(title="Pipeline")
    save_fig(out_dir / "summary_bar.png")


def plot_seq_boxplots(seq_csv: Path, out_dir: Path, normalize: bool = True) -> None:
    df = read_csv_clean(seq_csv)
    if df.empty:
        return

    metrics = ["mean_dice", "mean_iou", "mean_ciou", "centroid_disp_mean", "pred_area_std_norm"]
    df_plot = df.copy()
    if normalize:
        df_plot = normalize_columns(df_plot, metrics)
    for metric in metrics:
        ax = df_plot.boxplot(column=metric, by="pipeline", figsize=(6, 4))
        ax.set_title(f"Sequence Distribution: {metric}")
        ax.set_ylabel(metric)
        plt.suptitle("")
        save_fig(out_dir / f"seq_box_{metric}.png")


def plot_scatter_stability(seq_csv: Path, out_dir: Path, normalize: bool = True) -> None:
    df = read_csv_clean(seq_csv)
    if df.empty:
        return

    df_plot = df.copy()
    if normalize:
        df_plot = normalize_columns(df_plot, ["mean_iou", "centroid_disp_mean"])
    fig, ax = plt.subplots(figsize=(6, 5))
    for pipeline, group in df_plot.groupby("pipeline"):
        ax.scatter(group["mean_iou"], group["centroid_disp_mean"], label=pipeline, alpha=0.7)
    ax.set_xlabel("Mean IoU")
    ax.set_ylabel("Centroid Displacement Mean")
    ax.set_title("Accuracy vs Temporal Jitter (per sequence)")
    ax.legend()
    save_fig(out_dir / "scatter_iou_vs_jitter.png")


def plot_missing_counts(seq_csv: Path, out_dir: Path, normalize: bool = True) -> None:
    df = read_csv_clean(seq_csv)
    if df.empty:
        return

    df_plot = df.copy()
    agg = df_plot.groupby("pipeline")[ ["missing_pred_count", "missing_gt_count"] ].sum()
    if normalize:
        agg = normalize_columns(agg.reset_index(), ["missing_pred_count", "missing_gt_count"]).set_index("pipeline")
        ylabel = "Normalized Value"
    else:
        agg = agg
        ylabel = "Count"
    ax = agg.plot(kind="bar", figsize=(6, 4), rot=0)
    ax.set_title("Missing Frames (summed across sequences)")
    ax.set_ylabel(ylabel)
    save_fig(out_dir / "missing_counts.png")


def plot_frame_quality(frame_csv: Path, out_dir: Path, normalize: bool = True) -> None:
    df = read_csv_clean(frame_csv)
    if df.empty:
        return

    metrics = ["dice", "iou", "precision", "recall"]
    df_plot = df.copy()
    if normalize:
        df_plot = normalize_columns(df_plot, metrics)
    for metric in metrics:
        ax = df_plot.boxplot(column=metric, by="pipeline", figsize=(6, 4))
        ax.set_title(f"Frame-level Distribution: {metric}")
        ax.set_ylabel(metric)
        plt.suptitle("")
        save_fig(out_dir / f"frame_box_{metric}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize evaluation outputs")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("scripts/eval_outputs"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("scripts/eval_outputs/plots"),
    )
    args = parser.parse_args()
    # Default: do not normalize (plot raw values)
    normalize = False

    summary_csv = args.input_dir / "summary.csv"
    seq_csv = args.input_dir / "seq_metrics.csv"
    frame_csv = args.input_dir / "frame_metrics.csv"

    if summary_csv.exists():
        plot_summary(summary_csv, args.out_dir, normalize=normalize)
    if seq_csv.exists():
        plot_seq_boxplots(seq_csv, args.out_dir, normalize=normalize)
        plot_scatter_stability(seq_csv, args.out_dir, normalize=normalize)
        plot_missing_counts(seq_csv, args.out_dir, normalize=normalize)
    if frame_csv.exists():
        plot_frame_quality(frame_csv, args.out_dir, normalize=normalize)

    print(f"Plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
