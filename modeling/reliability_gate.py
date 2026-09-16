"""Write reliability for recurrent dynamic-token memory."""

from __future__ import annotations

import math


def token_write_reliability(
    mask_confidence: float, # SAM2 predicted IoU for the selected mask
    object_score: float, # SAM2 probability that the object is present
    has_foreground: bool, # whether the decoded mask contains any foreground pixel
    identity_similarity: float, # foreground-pooled feature consistency proxy
    area_plausibility: float,
    temporal_consistency: float,
) -> float:
    """Score foreground quality or empty-mask absence; this is not calibrated IoU."""
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
        return 1.0 - object_score
    signals = (
        mask_confidence,
        object_score,
        identity_similarity,
        area_plausibility,
        temporal_consistency,
    )
    return math.prod(signals) ** (1.0 / len(signals))
