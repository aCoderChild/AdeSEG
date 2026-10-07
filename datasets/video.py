"""Dataset-neutral, labelled-video clips for Kalman-memory training.

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

import cv2
import numpy as np
import torch

from adenoid.io import load_semantic_mask

IMAGE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class ManifestFrame:
    video_id: str
    frame_index: int
    image_path: Path
    mask_path: Path


def _resolve(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def load_manifest(manifest: Path, split: str) -> dict[str, list[ManifestFrame]]:
    """Load one split, rejecting malformed rows before training starts."""
    manifest = Path(manifest)
    videos: dict[str, list[ManifestFrame]] = defaultdict(list)
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
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
            raise FileNotFoundError(f"{manifest}:{line_number}: image or mask does not exist")
        videos[str(row["video_id"])].append(
            ManifestFrame(str(row["video_id"]), int(row["frame_index"]), image_path, mask_path)
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
            raise ValueError(f"A video appears in more than one split: {sorted(overlap)[:3]}")
        seen.update(videos)


def _semantic_mask(path: Path) -> np.ndarray:
    return load_semantic_mask(path)


class ManifestVideoClips:
    """Natural temporal clips for one requested semantic label at a time.

    A single Kalman update module is shared by every label.  Each sample remains
    binary because MedSAM2's prompt and the current Kalman state represent one
    object; multi-label data therefore trains the same update rule on each object
    without teaching it a dataset- or class-specific shortcut.
    """

    def __init__(
        self,
        videos: dict[str, list[ManifestFrame]],
        label_ids: list[int],
        image_size: int,
        clip_length: int,
    ):
        if clip_length < 2:
            raise ValueError("clip_length must be at least 2.")
        if not label_ids or any(label <= 0 for label in label_ids):
            raise ValueError("label_ids must be non-zero semantic-mask values.")
        self.image_size = image_size
        self.clip_length = clip_length
        self.clips: list[tuple[list[ManifestFrame], int]] = []
        seen_labels: set[int] = set()
        for frames in videos.values():
            if len(frames) < clip_length:
                continue
            labels_by_frame = [_semantic_mask(frame.mask_path) for frame in frames]
            for label_id in label_ids:
                present = [bool((mask == label_id).any()) for mask in labels_by_frame]
                if any(present):
                    seen_labels.add(label_id)
                for start in range(len(frames) - clip_length + 1):
                    if present[start]:
                        self.clips.append((frames[start : start + clip_length], label_id))
        missing = sorted(set(label_ids) - seen_labels)
        if missing:
            raise ValueError(f"Requested label_ids are absent from this split: {missing}")
        if not self.clips:
            raise ValueError("No clip has a prompted object followed by enough annotated frames.")

    def __len__(self) -> int:
        return len(self.clips)

    def _image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    def _mask(self, path: Path, label_id: int) -> np.ndarray:
        mask = (_semantic_mask(path) == label_id).astype(np.uint8)
        return cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

    def sample(self, rng: np.random.Generator) -> dict[str, torch.Tensor | str | int]:
        frames, label_id = self.clips[int(rng.integers(len(self.clips)))]
        images = np.stack([self._image(frame.image_path) for frame in frames]).astype(np.float32) / 255.0
        masks = np.stack([self._mask(frame.mask_path, label_id) for frame in frames])
        return {
            "images": torch.from_numpy((images - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2).float(),
            "masks": torch.from_numpy(masks.astype(np.float32)),
            "frame_gaps": torch.tensor(
                [0, *[right.frame_index - left.frame_index for left, right in zip(frames, frames[1:])]],
                dtype=torch.float32,
            ),
            "present": torch.from_numpy(masks.reshape(len(masks), -1).any(axis=1)),
            "source": str(frames[0].image_path),
            "video_id": frames[0].video_id,
            "label_id": label_id,
            "absence_mode": "natural",
        }
