"""Frame discovery, image decoding, and box-prompt preprocessing."""

from __future__ import annotations

import re
from pathlib import Path

from adenoid.dataset import FrameSample
from adenoid.io import load_binary_mask, resolve_mask_path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG")


def get_numeric_sort_key(name: str):
    parts = re.split(r"(\d+)", name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def is_image_file(path: str) -> bool:
    return Path(path).suffix in IMAGE_EXTENSIONS


def get_video_frame_dir(base_video_dir: str | Path, video_name: str) -> str:
    base_video_dir = Path(base_video_dir)
    if video_name == ".":
        images = base_video_dir / "images"
        return str(images if images.is_dir() else base_video_dir)
    images = base_video_dir / video_name / "images"
    if images.is_dir():
        return str(images)
    polypgen_images = base_video_dir / video_name / f"images_{video_name}"
    return str(polypgen_images if polypgen_images.is_dir() else base_video_dir / video_name)


def get_video_name(base_video_dir: str | Path, video_name: str) -> str:
    if video_name != ".":
        return video_name
    base = Path(base_video_dir).resolve()
    return base.parent.name if base.name == "images" else base.name


def list_video_names(base_video_dir: str | Path) -> list[str]:
    base = Path(base_video_dir)
    images = base / "images"
    if images.is_dir() and any(is_image_file(path.name) for path in images.iterdir()):
        return ["."]
    videos = [
        path.name
        for path in base.iterdir()
        if path.is_dir()
        and any(
            is_image_file(frame_path.name)
            for frame_path in Path(get_video_frame_dir(base, path.name)).iterdir()
        )
    ]
    if videos:
        return sorted(videos, key=get_numeric_sort_key)
    return ["."] if any(is_image_file(path.name) for path in base.iterdir()) else []


def get_frame_names(video_dir: str | Path) -> list[str]:
    return sorted(
        (path.stem for path in Path(video_dir).iterdir() if is_image_file(path.name)),
        key=get_numeric_sort_key,
    )


def resolve_frame_path(video_dir: str | Path, frame_name: str) -> str:
    for extension in IMAGE_EXTENSIONS:
        path = Path(video_dir) / f"{frame_name}{extension}"
        if path.exists():
            return str(path)
    raise FileNotFoundError(Path(video_dir) / f"{frame_name}.jpg")


def select_video_names(base_video_dir: str | Path, seq_nums=None, video_list_file: str | Path | None = None) -> list[str]:
    if seq_nums:
        return [f"seq{sequence_number}" for sequence_number in seq_nums]
    if video_list_file:
        return [line.strip() for line in Path(video_list_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    return list_video_names(base_video_dir)


def iter_polypgen_samples(data_root: str | Path, sequence_name: str):
    """Yield a PolypGen sequence as common frame samples with a ``polyp`` mask."""
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
