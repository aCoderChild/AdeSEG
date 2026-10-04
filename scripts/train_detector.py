#!/usr/bin/env python3
"""Train the YOLOv8n polyp detector on the dataset from build_detector_dataset.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="polyp.yaml written by build_detector_dataset.py")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="detector_yolov8n")
    parser.add_argument("--weights", default="yolov8n.pt", help="COCO-pretrained start point")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    YOLO(args.weights).train(
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        deterministic=False,
        patience=20,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
