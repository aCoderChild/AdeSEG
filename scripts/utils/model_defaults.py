"""Shared model defaults for controlled SAM/YOLO experiments."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SAM2_CODE_ROOT = os.environ.get(
    "SAM2_CODE_ROOT",
    str(PROJECT_ROOT / "external" / "MedSAM2"),
)
DEFAULT_SAM2_CFG = os.environ.get(
    "SAM2_CFG",
    "configs/sam2.1_hiera_t512.yaml",
)
DEFAULT_SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    str(PROJECT_ROOT / "checkpoints" / "MedSAM2_latest.pt"),
)
DEFAULT_YOLO_CHECKPOINT = os.environ.get(
    "YOLO_CHECKPOINT",
    str(PROJECT_ROOT / "checkpoints" / "polypgen_yolov8n.pt"),
)
