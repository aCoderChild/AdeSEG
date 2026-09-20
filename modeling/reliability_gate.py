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
    detector_geometry_agreement: float | None = None,
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
    if detector_geometry_agreement is not None:
        if not math.isfinite(detector_geometry_agreement) or not 0.0 <= detector_geometry_agreement <= 1.0:
            raise ValueError("detector_geometry_agreement must be finite in [0, 1].")
        signals = (*signals, detector_geometry_agreement)
    return math.prod(signals) ** (1.0 / len(signals))


def reliability_gated_ema_weight(
    reliability: float,
    minimum_weight: float,
    maximum_weight: float,
    reliability_power: float,
) -> float:
    """Map a reliability score to the EMA coefficient for a gated write."""
    if not 0.0 <= reliability <= 1.0:
        raise ValueError("reliability must be in [0, 1].")
    if not 0.0 <= minimum_weight <= maximum_weight <= 1.0:
        raise ValueError("EMA weights must satisfy 0 <= minimum <= maximum <= 1.")
    if reliability_power <= 0.0:
        raise ValueError("reliability_power must be positive.")
    return minimum_weight + (maximum_weight - minimum_weight) * (
        reliability ** reliability_power
    )
