"""Shared binary-mask I/O helpers for AdeSEG experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG")


def resolve_mask_path(mask_dir: Path, frame_name: str) -> Path | None:
    """Resolve a frame-aligned mask across supported image extensions."""
    for extension in MASK_EXTENSIONS:
        for stem in (frame_name, f"{frame_name}_mask"):
            candidate = mask_dir / f"{stem}{extension}"
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


def save_ann_png(path: str | Path, mask: np.ndarray, palette: bytes) -> None:
    """Save an indexed uint8 mask with the supplied PNG palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def put_per_obj_mask(
    per_obj_mask: dict[int, np.ndarray], height: int, width: int
) -> np.ndarray:
    """Combine per-object masks into one indexed uint8 mask.

    Object IDs are applied in descending order, preserving the conflict
    resolution previously used by the video-inference experiments.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for object_id in sorted(per_obj_mask)[::-1]:
        object_mask = np.asarray(per_obj_mask[object_id], dtype=bool).reshape(height, width)
        mask[object_mask] = object_id
    return mask


def save_palette_masks_to_dir(
    output_mask_dir: str | Path,
    video_name: str,
    frame_name: str,
    per_obj_output_mask: dict[int, np.ndarray],
    height: int,
    width: int,
    per_obj_png_file: bool,
    output_palette: bytes,
) -> None:
    """Save combined or per-object indexed masks beneath a video directory."""
    video_output_dir = Path(output_mask_dir) / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        save_ann_png(video_output_dir / f"{frame_name}.png", output_mask, output_palette)
        return

    for object_id, object_mask in per_obj_output_mask.items():
        object_output_dir = video_output_dir / f"{object_id:03d}"
        object_output_dir.mkdir(parents=True, exist_ok=True)
        output_mask = object_mask.reshape(height, width).astype(np.uint8)
        save_ann_png(object_output_dir / f"{frame_name}.png", output_mask, output_palette)


def save_masks_to_dir(
    output_mask_dir: str | Path,
    video_name: str,
    frame_name: str,
    per_obj_output_mask: dict[int, np.ndarray],
    height: int,
    width: int,
    per_obj_png_file: bool,
) -> None:
    """Save combined or per-object grayscale masks beneath a video directory."""
    video_output_dir = Path(output_mask_dir) / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        assert output_mask.dtype == np.uint8
        assert output_mask.ndim == 2
        Image.fromarray(output_mask).save(video_output_dir / f"{frame_name}.png")
        return

    for object_id, object_mask in per_obj_output_mask.items():
        object_output_dir = video_output_dir / f"{object_id:03d}"
        object_output_dir.mkdir(parents=True, exist_ok=True)
        output_mask = object_mask.reshape(height, width).astype(np.uint8)
        assert output_mask.dtype == np.uint8
        assert output_mask.ndim == 2
        Image.fromarray(output_mask).save(object_output_dir / f"{frame_name}.png")


def make_overlay(
    frame_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend ground-truth (green) and predicted (red) masks over a frame.

    Pixels where both masks agree come out yellow (green+red), since the two
    color layers are additive before blending.
    """
    color_layer = np.zeros_like(frame_rgb, dtype=np.float32)
    color_layer[..., 1] = np.where(gt_mask > 0, 255, 0)
    color_layer[..., 0] = np.where(pred_mask > 0, 255, 0)

    covered = (gt_mask > 0) | (pred_mask > 0)
    overlay = frame_rgb.astype(np.float32).copy()
    overlay[covered] = (
        overlay[covered] * (1 - alpha) + color_layer[covered] * alpha
    )
    return overlay.astype(np.uint8)


def draw_box(
    overlay_rgb: np.ndarray,
    box: np.ndarray | None,
    color: tuple[int, int, int] = (0, 255, 255),
    width: int = 2,
) -> np.ndarray:
    """Draw the predicted (x1, y1, x2, y2) box outline onto an overlay image."""
    if box is None:
        return overlay_rgb
    image = Image.fromarray(overlay_rgb)
    ImageDraw.Draw(image).rectangle([float(v) for v in box], outline=color, width=width)
    return np.array(image)


def save_overlay(overlay_rgb: np.ndarray, overlay_path: Path) -> None:
    """Save an RGB overlay image (see `make_overlay`) as a PNG."""
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay_rgb).save(overlay_path)
