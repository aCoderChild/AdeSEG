"""Load frozen MedSAM2 and YOLO models for the inference scripts."""

from __future__ import annotations


def load_yolo_model(checkpoint: str):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("YOLO inference requires the 'ultralytics' package.") from error
    return YOLO(str(checkpoint))


def build_image_predictor(
    config_file: str,
    checkpoint: str,
    device: str | None = None,
    apply_postprocessing: bool = False,
):
    from MedSAM2.sam2.build_sam import build_sam2
    from MedSAM2.sam2.sam2_image_predictor import SAM2ImagePredictor

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
    from MedSAM2.sam2.build_sam import build_sam2_video_predictor

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
