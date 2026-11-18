"""Verified checkpoint loader for the *separate* MedSAM2 Hiera-T/512 study.

This is not the 0.7421 Lean-SAM2 Hiera-B+ benchmark. A filename alone
does not establish weight identity; loading checks the complete SHA-256.
The exact model configuration must be explicitly specified by the caller.
"""
import hashlib
from pathlib import Path

from modeling.medsam2 import build_video_predictor

MEDSAM2_LATEST_SHA256 = "c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060"


def verify_medsam2_latest(checkpoint):
    """Reject missing or different checkpoint files before model creation."""
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MEDSAM2_LATEST_SHA256:
        raise ValueError(
            "Checkpoint mismatch: expected recorded MedSAM2_latest.pt SHA-256 "
            + MEDSAM2_LATEST_SHA256
        )
    return checkpoint


def build_verified_medsam2_latest(checkpoint, *, model_cfg, device="auto",
                                  apply_postprocessing=False):
    """Build native MedSAM2 with explicitly selected, matching Hiera-T config.

    The config string must match the recorded experiment's model package; it
    is intentionally not inferred from 'latest' in the checkpoint filename.
    This does NOT install the Lean-SAM2 bypass, memory pruner or RoPE patch.
    """
    if not isinstance(model_cfg, str) or not model_cfg.strip():
        raise ValueError("Explicit MedSAM2 Hiera-T configuration is required")
    return build_video_predictor(
        model_cfg, verify_medsam2_latest(checkpoint),
        device=device, apply_postprocessing=apply_postprocessing)
