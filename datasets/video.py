"""Manifest loading and ordered frame staging for native video inference.

The manifest is JSON Lines.  Each row describes one annotated frame:
``{"split": "train", "video_id": "case_01", "frame_index": 0,
"image": "frames/case_01/000000.jpg", "mask": "masks/case_01/000000.png"}``.
Images and masks are resolved relative to the manifest.  Masks are indexed: 0 is
background and every requested ``label_id`` is sampled as its own binary object.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IMAGE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class ManifestFrame:
    video_id: str
    frame_index: int
    image_path: Path
    mask_path: Path


def stage_frames(frames: list[ManifestFrame], directory: Path) -> None:
    """Link manifest-ordered frames to SAM2's contiguous JPEG input names."""
    for index, frame in enumerate(frames):
        (directory / f"{index:06d}.jpg").symlink_to(frame.image_path.resolve())


def _resolve(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def load_manifest(manifest: Path, split: str) -> dict[str, list[ManifestFrame]]:
    """Load one split, rejecting malformed rows before training starts."""
    manifest = Path(manifest)
    videos: dict[str, list[ManifestFrame]] = defaultdict(list)
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{manifest}:{line_number}: invalid JSON") from error
        if row.get("split") != split or not row.get("valid", True):
            continue
        missing = {"video_id", "frame_index", "image", "mask"} - row.keys()
        if missing:
            raise ValueError(f"{manifest}:{line_number}: missing {sorted(missing)}")
        image_path = _resolve(str(row["image"]), manifest)
        mask_path = _resolve(str(row["mask"]), manifest)
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"{manifest}:{line_number}: image or mask does not exist"
            )
        videos[str(row["video_id"])].append(
            ManifestFrame(
                str(row["video_id"]), int(row["frame_index"]), image_path, mask_path
            )
        )
    if not videos:
        raise ValueError(f"No valid frames for split {split!r} in {manifest}.")
    for video_id, frames in videos.items():
        frames.sort(key=lambda frame: frame.frame_index)
        if len({frame.frame_index for frame in frames}) != len(frames):
            raise ValueError(f"Duplicate frame_index values in video {video_id!r}.")
    return dict(videos)


def assert_disjoint_videos(*splits: dict[str, list[ManifestFrame]]) -> None:
    seen: set[str] = set()
    for videos in splits:
        overlap = seen & videos.keys()
        if overlap:
            raise ValueError(
                f"A video appears in more than one split: {sorted(overlap)[:3]}"
            )
        seen.update(videos)
