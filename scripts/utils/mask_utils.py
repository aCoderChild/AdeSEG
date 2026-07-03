"""Shared binary-mask I/O helpers for AdeSEG experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG")


def resolve_mask_path(mask_dir: Path, frame_name: str) -> Path | None:
    """Resolve a frame-aligned mask across supported image extensions."""
    for extension in MASK_EXTENSIONS:
        candidate = mask_dir / f"{frame_name}{extension}"
        if candidate.exists():
            return candidate
    return None


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load a grayscale mask as uint8 values in {0, 1}.

    JPEG masks are thresholded at 127 to avoid treating compression noise as
    foreground. Native 0/1 masks are thresholded at zero.
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    threshold = 0 if mask.max() <= 1 else 127
    return (mask > threshold).astype(np.uint8)


def resize_binary_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask with nearest-neighbor interpolation."""
    target_h, target_w = shape_hw
    if mask.shape == (target_h, target_w):
        return (mask > 0).astype(np.uint8)
    pil_mask = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    resized = pil_mask.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
    return (np.array(resized) > 0).astype(np.uint8)


def save_binary_mask(mask: np.ndarray, mask_path: Path) -> None:
    """Save a binary mask as a 0/255 grayscale PNG-compatible image."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(mask_path)


def list_mask_frame_names(
    mask_dir: Path,
    sort_key,
    extensions: Iterable[str] = MASK_EXTENSIONS,
) -> list[str]:
    """List unique frame stems in a mask directory using a supplied sort key."""
    frame_names = []
    for extension in extensions:
        frame_names.extend(path.stem for path in mask_dir.glob(f"*{extension}"))
    return sorted(set(frame_names), key=sort_key)
