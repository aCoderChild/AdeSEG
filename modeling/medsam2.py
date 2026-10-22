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
    apply_postprocessing=True,
):
    """Build the bundled native video predictor; disable machine-dependent hole filling."""
    from pathlib import Path
    import sys

    # Hydra resolves the bundled author's ``sam2`` config package. Do not rely
    # on a retired experiment runner to install this import path as a side effect.
    native_root = str(Path(__file__).resolve().parents[1] / "external/MedSAM2")
    if native_root not in sys.path:
        sys.path.insert(0, native_root)
    from external.MedSAM2.sam2.build_sam import build_sam2_video_predictor

    device = _resolve_device(device)
    model = build_sam2_video_predictor(
        model_cfg,
        str(checkpoint),
        device=device,
        apply_postprocessing=apply_postprocessing,
    )
    # MedSAM2 fills small mask holes only when its compiled _C extension is installed, so
    # outputs would depend on the machine; every reported run had no hole filling.
    model.fill_hole_area = 0
    return model
