"""Thin MedSAM2/SAM2 construction and prompt-detection helpers."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


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
    """Build native MedSAM2 or a custom video predictor."""
    from MedSAM2.sam2.build_sam import build_sam2_video_predictor

    device = _resolve_device(device)
    if predictor_target is None:
        return build_sam2_video_predictor(
            model_cfg,
            str(checkpoint),
            device=device,
        )

    module_name, class_name = str(predictor_target).rsplit(".", 1)
    predictor_class = getattr(importlib.import_module(module_name), class_name)
    overrides = dict(predictor_overrides or {})
    return build_sam2_video_predictor(
        model_cfg,
        str(checkpoint),
        device=device,
        vos_optimized=False,
        _target_=predictor_class,
        **overrides,
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
