"""Write reliability for recurrent dynamic-token memory.

Runtime hyperparameters are supplied by the inference config, not this gate.
"""

from __future__ import annotations

import math


def token_write_reliability(
    mask_confidence: float, # SAM2 predicted IoU for the selected mask
    object_score: float, # SAM2 probability that the object is present
    has_foreground: bool, # whether the decoded mask contains any foreground pixel
    identity_similarity: float, # foreground-pooled feature consistency proxy
    area_plausibility: float,
    temporal_consistency: float,
    background_confirmed: bool = False,
    unconfirmed_absence_scale: float = 0.25,
) -> float:
    """Return a continuous write score from segmentation and external evidence."""
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (
            mask_confidence,
            object_score,
            identity_similarity,
            area_plausibility,
            temporal_consistency,
        )
    ):
        raise ValueError(
            "Token reliability inputs must be finite in [0, 1]."
        )
    if not has_foreground:
        if not 0.0 <= unconfirmed_absence_scale <= 1.0:
            raise ValueError("unconfirmed_absence_scale must be in [0, 1].")
        absence_scale = 1.0 if background_confirmed else unconfirmed_absence_scale
        return (1.0 - object_score) * absence_scale
    signals = (
        mask_confidence,
        object_score,
        identity_similarity,
        area_plausibility,
        temporal_consistency,
    )
    return math.prod(signals) ** (1.0 / len(signals))
