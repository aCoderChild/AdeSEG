"""Thin MedSAM2/SAM2 construction and prompt-detection helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_video_frame_like_predictor(frame_path, image_size):
    """Load one frame with the exact resize and normalization used by video inference."""
    import torch
    from PIL import Image

    image = Image.open(frame_path)
    pixels = np.array(image.convert("RGB").resize((image_size, image_size)))
    if pixels.dtype != np.uint8:
        raise RuntimeError(f"Expected uint8 image data in {frame_path}.")
    tensor = torch.from_numpy(pixels / 255.0).permute(2, 0, 1).float()
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def _resolve_device(device: str):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def build_video_predictor(
    model_cfg,
    checkpoint,
    device="auto",
    predictor_target=None,
    predictor_overrides=None,
):
    """Build native MedSAM2 or a custom video predictor.

    Custom predictors are selected through Hydra configuration overrides so the
    bundled MedSAM2 builder remains the single construction path.
    """
    from MedSAM2.sam2.build_sam import build_sam2_video_predictor

    device = _resolve_device(device)
    hydra_overrides = []
    if predictor_target is not None:
        hydra_overrides.append(f"++model._target_={predictor_target}")
        for key, value in (predictor_overrides or {}).items():
            if isinstance(value, bool):
                value = str(value).lower()
            hydra_overrides.append(f"++model.{key}={value}")
    return build_sam2_video_predictor(
        model_cfg,
        str(checkpoint),
        device=device,
        hydra_overrides_extra=hydra_overrides,
    )


def build_image_predictor(model_cfg, checkpoint, device="auto"):
    """Build a MedSAM2 image predictor from the bundled SAM2 implementation."""
    from MedSAM2.sam2.build_sam import build_sam2
    from MedSAM2.sam2.sam2_image_predictor import SAM2ImagePredictor

    device = _resolve_device(device)
    model = build_sam2(model_cfg, str(checkpoint), device=device)
    return SAM2ImagePredictor(model)


def load_yolo_model(checkpoint):
    from ultralytics import YOLO

    return YOLO(str(checkpoint))


def get_yolo_boxes(model, frame_path, image_size=640, confidence=0.5, max_boxes=1):
    """Return highest-confidence YOLO xyxy boxes and confidences."""
    results = model.predict(
        source=str(frame_path),
        imgsz=image_size,
        conf=confidence,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []
    boxes = results[0].boxes
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = boxes.conf.detach().cpu().numpy().astype(float)
    order = np.argsort(-scores)
    if max_boxes is not None:
        order = order[:max_boxes]
    return [(xyxy[index], float(scores[index])) for index in order]
