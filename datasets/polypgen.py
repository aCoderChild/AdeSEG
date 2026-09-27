"""PolypGen adapter for the common ``FrameSample`` format."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from inference.data import get_frame_names, get_video_frame_dir, resolve_frame_path
from utils.mask_utils import load_binary_mask, resolve_mask_path

from .base_dataset import FrameSample


def iter_polypgen_samples(data_root: str | Path, sequence_name: str) -> Iterator[FrameSample]:
    """Yield a PolypGen sequence as frames with one named ``polyp`` mask."""
    data_root = Path(data_root)
    video_dir = Path(get_video_frame_dir(data_root, sequence_name))
    sequence_number = sequence_name.removeprefix("seq")
    mask_dir = data_root / sequence_name / f"masks_seq{sequence_number}"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing PolypGen masks: {mask_dir}")

    for frame_idx, frame_name in enumerate(get_frame_names(video_dir)):
        mask_path = resolve_mask_path(mask_dir, frame_name)
        if mask_path is None:
            raise FileNotFoundError(f"Missing PolypGen mask for {sequence_name}/{frame_name}")
        yield FrameSample(
            frame_idx=frame_idx,
            image_path=Path(resolve_frame_path(video_dir, frame_name)),
            masks={"polyp": load_binary_mask(mask_path)},
            metadata={"sequence": sequence_name, "frame_name": frame_name},
        )
