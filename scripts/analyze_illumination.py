#!/usr/bin/env python3
"""Illumination-stratified Dice. Joins per-frame CSVs with per-frame luminance
and specular-highlight fraction computed directly from PolypGen images.

No re-inference. Reads existing run outputs under ``metrics_per_frame.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
from scipy.stats import wilcoxon


DEV_SEQUENCES = {f"seq{i}" for i in range(1, 16)}
TEST_SEQUENCES = {f"seq{i}" for i in range(16, 24)}


def frame_image(polypgen_root: Path, sequence: str, frame_stem: str) -> Path:
    index = int(sequence[3:])
    return polypgen_root / "sequenceData" / "positive" / sequence / f"images_seq{index}" / f"{frame_stem}.jpg"


def image_stats(image: np.ndarray) -> tuple[float, float]:
    """Return mean luminance (0..1) and specular-highlight fraction (pixels > 0.95)."""
    luma = (0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]) / 255.0
    return float(luma.mean()), float((luma > 0.95).mean())


def build_frame_stats(polypgen_root: Path, sequences, cache: Path):
    if cache.exists():
        with cache.open() as fh:
            return json.load(fh)
    stats = {}
    for sequence in sorted(sequences):
        index = int(sequence[3:])
        folder = polypgen_root / "sequenceData" / "positive" / sequence / f"images_seq{index}"
        if not folder.exists():
            continue
        for file in sorted(folder.glob("*.jpg")):
            image = np.asarray(Image.open(file).convert("RGB"))
            luma, specular = image_stats(image)
            stats[f"{sequence}|{file.stem}"] = {"luma": luma, "specular": specular}
        print(f"{sequence}: {len([k for k in stats if k.startswith(sequence + '|')])} frames", flush=True)
    cache.write_text(json.dumps(stats))
    return stats


def load_rows(path: Path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def paired(diffs):
    values = np.asarray(list(diffs.values()))
    if len(values) < 3:
        return None
    try:
        _, p = wilcoxon(values)
    except ValueError:
        p = float("nan")
    return {"n": len(values), "mean": float(values.mean()), "p": float(p),
            "better": int((values > 0).sum()), "worse": int((values < 0).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polypgen_root", type=Path,
                        default=Path("data/PolypGen2021_MultiCenterData_v3"))
    parser.add_argument("--analysis_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--terciles", type=float, nargs=2, default=None,
                        help="Luminance terciles; if omitted, compute from the selected sequences.")
    args = parser.parse_args()

    sequences = {"dev": DEV_SEQUENCES, "test": TEST_SEQUENCES,
                 "all": DEV_SEQUENCES | TEST_SEQUENCES}[args.split]

    cache = args.analysis_dir / f"frame_stats_{args.split}.json"
    stats = build_frame_stats(args.polypgen_root, sequences, cache)

    runs = {
        "native": "01_native_medsam2.csv",
        "native_k1": "02_native_medsam2_prompt_plus_1frame.csv",
        "kalman": "10_kalman_memory.csv",
        "ema": "19_kalman_fixed_gain_ema.csv",
    }

    # Compute terciles over polyp frames in the selected sequences, from the native CSV.
    native_rows = [r for r in load_rows(args.analysis_dir / runs["native"])
                   if r["sequence"] in sequences and r["gt_present"].lower() == "true"]
    luminas = []
    for row in native_rows:
        key = f"{row['sequence']}|{row['frame']}"
        if key in stats:
            luminas.append(stats[key]["luma"])
    luminas = np.asarray(luminas)
    if args.terciles is None:
        q1, q2 = float(np.quantile(luminas, 1 / 3)), float(np.quantile(luminas, 2 / 3))
    else:
        q1, q2 = args.terciles

    def bucket(key):
        if key not in stats:
            return None
        luma = stats[key]["luma"]
        if luma <= q1:
            return "low"
        if luma <= q2:
            return "mid"
        return "high"

    print(f"Luminance terciles over {args.split} polyp frames: low <= {q1:.3f} < mid <= {q2:.3f}\n")

    seq_by_bucket = {name: defaultdict(lambda: defaultdict(list)) for name in runs}  # name -> bucket -> seq -> dices
    for name, filename in runs.items():
        for row in load_rows(args.analysis_dir / filename):
            if row["sequence"] not in sequences:
                continue
            if row["gt_present"].lower() != "true":
                continue
            if row["frames_after_prompt"] in ("", "nan"):  # prompt frame
                continue
            b = bucket(f"{row['sequence']}|{row['frame']}")
            if b is None:
                continue
            seq_by_bucket[name][b][row["sequence"]].append(float(row["dice"]))

    print(f"{'bucket':>8}  " + "  ".join(f"{n:>10}" for n in runs))
    for b in ("low", "mid", "high"):
        line = [f"{b:>8}"]
        for name in runs:
            seq_dice = {s: mean(vs) for s, vs in seq_by_bucket[name][b].items()}
            line.append(f"{mean(seq_dice.values()):>10.3f}" if seq_dice else f"{'nan':>10}")
        print("  ".join(line))

    print("\nPaired tests per bucket (vs native):")
    for b in ("low", "mid", "high"):
        for other in ("native_k1", "kalman", "ema"):
            native = {s: mean(vs) for s, vs in seq_by_bucket["native"][b].items()}
            comp = {s: mean(vs) for s, vs in seq_by_bucket[other][b].items()}
            shared = native.keys() & comp.keys()
            diffs = {s: comp[s] - native[s] for s in shared}
            r = paired(diffs)
            if r:
                print(f"  {b:>5}  {other:>12}: n={r['n']}  diff {r['mean']:+.3f}  "
                      f"{r['better']}/{r['worse']}  p={r['p']:.3f}")


if __name__ == "__main__":
    main()
