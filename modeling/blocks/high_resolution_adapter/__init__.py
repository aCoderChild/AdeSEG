"""High-resolution decoder-skip adapter with original five-layer weight names.

This is the new reconstruction, not a verified historical forward replay.
"""
from modeling.medical_highres.adapter import MedicalHighResolutionAdapter
from modeling.medical_highres.integration import attach_medical_highres_adapter
from modeling.medical_highres.blocks import (
    CrossScaleProjection,
    DepthwiseSpatialFusion,
    BoundedSkipResidual,
)

__all__ = [
    "MedicalHighResolutionAdapter",
    "attach_medical_highres_adapter",
    "CrossScaleProjection",
    "DepthwiseSpatialFusion",
    "BoundedSkipResidual",
]
