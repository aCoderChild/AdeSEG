#!/usr/bin/env python3
"""Run MedSAM2 zero-shot REFUGE2 inference with oracle GT-box prompts."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "MedSAM2"))
from inference.image import run_refuge2_oracle_boxes
from modeling.medsam2 import build_image_predictor

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=ROOT / "configs/refuge2.yaml")
parser.add_argument("--split", default="val", choices=("train", "val", "test"))
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--limit", type=int)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--device", default=None)
args = parser.parse_args()
config = json.loads(args.config.read_text())
predictor = build_image_predictor(config["sam2_cfg"], ROOT / config["sam2_checkpoint"], args.device or config["device"])
run_refuge2_oracle_boxes(predictor, ROOT / config["data_root"], args.split, args.output_dir, args.start, args.limit)
