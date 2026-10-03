#!/usr/bin/env python3
"""Run MedSAM2 zero-shot REFUGE2 inference with oracle GT-box prompts."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MedSAM2"))

from evaluation.refuge2 import evaluate_refuge2
from inference.image import run_refuge2_oracle_boxes
from modeling.medsam2 import build_image_predictor


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=ROOT / "configs/refuge2.yaml")
parser.add_argument("--split", default="val", choices=("train", "val", "test"))
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--limit", type=int)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--device", default=None)
parser.add_argument("--evaluate_only", action="store_true")
args = parser.parse_args()

config = json.loads(args.config.read_text())
if not args.evaluate_only:
    predictor = build_image_predictor(
        config["sam2_cfg"],
        ROOT / config["sam2_checkpoint"],
        args.device or config["device"],
    )
    run_refuge2_oracle_boxes(
        predictor,
        ROOT / config["data_root"],
        args.split,
        args.output_dir,
        args.start,
        args.limit,
    )

if args.limit is None:
    metrics = evaluate_refuge2(
        args.output_dir,
        ROOT / config["data_root"],
        args.split,
        args.output_dir / "metrics_per_image.csv",
    )
    summary = {
        "protocol": (
            "MedSAM2 zero-shot + GT-box oracle; disc/cup/rim segmentation "
            "with vCDR and area-ratio measurements"
        ),
        "split": args.split,
        **metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
