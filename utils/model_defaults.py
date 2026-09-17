"""Backward-compatible aliases for ``configs/model_config/frame_configs.json``."""

import os

from inference.config import (
    DEFAULT_FRAME_CONFIG,
    PROJECT_ROOT,
    load_json_config,
    resolve_project_path,
)


_DEFAULTS = load_json_config(DEFAULT_FRAME_CONFIG)
DEFAULT_SAM2_CODE_ROOT = os.environ.get("SAM2_CODE_ROOT", str(PROJECT_ROOT / "MedSAM2"))
DEFAULT_SAM2_CFG = os.environ.get("SAM2_CFG", _DEFAULTS["sam2_cfg"])
DEFAULT_SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT", str(resolve_project_path(_DEFAULTS["sam2_checkpoint"]))
)
DEFAULT_YOLO_CHECKPOINT = os.environ.get(
    "YOLO_CHECKPOINT", str(resolve_project_path(_DEFAULTS["yolo_checkpoint"]))
)
