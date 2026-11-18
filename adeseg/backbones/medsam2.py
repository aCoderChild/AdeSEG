"""MedSAM2-latest (Hiera-T/512) model construction and checkpoint verification.

This is NOT the SAM2.1 Hiera-B+ checkpoint of the 0.7421 Lean baseline.
"""
from modeling.medsam2_latest import (
    MEDSAM2_LATEST_SHA256,
    verify_medsam2_latest,
    build_verified_medsam2_latest,
)

__all__ = [
    "MEDSAM2_LATEST_SHA256",
    "verify_medsam2_latest",
    "build_verified_medsam2_latest",
]
