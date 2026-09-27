"""Load frozen MedSAM2 and YOLO models for the inference scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_yolo_model(checkpoint: str):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("YOLO inference requires the 'ultralytics' package.") from error
    return YOLO(str(checkpoint))


def get_yolo_boxes(
    yolo_model, frame_path: str | Path, yolo_imgsz: int, yolo_conf: float, max_boxes: int
):
    """Return highest-confidence YOLO xyxy boxes for one frame."""
    with Image.open(frame_path) as image:
        results = yolo_model.predict([image.convert("RGB")], imgsz=yolo_imgsz, conf=yolo_conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    order = np.argsort(confidence)[::-1]
    if max_boxes > 0:
        order = order[:max_boxes]
    return [(xyxy[index].astype(np.float32), float(confidence[index])) for index in order]


def build_image_predictor(
    config_file: str,
    checkpoint: str,
    device: str | None = None,
    apply_postprocessing: bool = False,
):
    from MedSAM2.sam2.build_sam import build_sam2, get_best_available_device
    from MedSAM2.sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "auto":
        device = get_best_available_device()
    model = build_sam2(
        config_file=config_file,
        ckpt_path=str(checkpoint),
        device=device,
        apply_postprocessing=apply_postprocessing,
    )
    return SAM2ImagePredictor(model)


def build_video_predictor(
    config_file: str,
    checkpoint: str,
    device: str | None = None,
    apply_postprocessing: bool = False,
    hydra_overrides_extra=None,
    vos_optimized: bool = False,
):
    from MedSAM2.sam2.build_sam import build_sam2_video_predictor, get_best_available_device

    if device == "auto":
        device = get_best_available_device()

    kwargs = {
        "config_file": config_file,
        "ckpt_path": str(checkpoint),
        "apply_postprocessing": apply_postprocessing,
        "hydra_overrides_extra": hydra_overrides_extra,
        "vos_optimized": vos_optimized,
    }
    if device is not None:
        kwargs["device"] = device
    return build_sam2_video_predictor(**kwargs)
