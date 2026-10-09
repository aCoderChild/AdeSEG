"""Thin MedSAM2/SAM2 video-predictor construction."""

from __future__ import annotations


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
    from external.MedSAM2.sam2.build_sam import build_sam2_video_predictor

    device = _resolve_device(device)
    hydra_overrides = []
    if predictor_target is not None:
        hydra_overrides.append(f"++model._target_={predictor_target}")
        for key, value in (predictor_overrides or {}).items():
            if isinstance(value, bool):
                value = str(value).lower()
            hydra_overrides.append(f"++model.{key}={value}")
    model = build_sam2_video_predictor(
        model_cfg,
        str(checkpoint),
        device=device,
        hydra_overrides_extra=hydra_overrides,
    )
    # MedSAM2 fills small mask holes only when its compiled _C extension is installed, so
    # outputs would depend on the machine; every reported run had no hole filling.
    model.fill_hole_area = 0
    return model
