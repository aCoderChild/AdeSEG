"""Small JSON configuration helpers for inference scripts."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
DEFAULT_DATA_CONFIG = CONFIG_ROOT / "data_config.json"
DEFAULT_FRAME_CONFIG = CONFIG_ROOT / "model_config" / "frame_configs.json"
DEFAULT_VIDEO_CONFIG = CONFIG_ROOT / "model_config" / "video_configs.json"
DEFAULT_DYNAMIC_CONFIG = CONFIG_ROOT / "model_config" / "dynamic_configs.json"


def load_json_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
