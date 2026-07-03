"""Reliability scoring and output-state gating for video mask experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass(frozen=True)
class ReliabilityConfig:
    confidence_weight: float = 0.35
    temporal_weight: float = 0.30
    area_weight: float = 0.25
    blur_weight: float = 0.10
    use_blur_score: bool = True
    blur_reference: float = 150.0
    min_mask_area_ratio: float = 0.0005
    max_mask_area_ratio: float = 0.80
    blank_mask_penalty: float = 0.25 # penalties
    implausible_area_penalty: float = 0.50 # penalties
    area_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        weights = (
            self.confidence_weight,
            self.temporal_weight,
            self.area_weight,
            self.blur_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Reliability weights must be non-negative")
        active_weight = sum(weights[:3]) + (
            self.blur_weight if self.use_blur_score else 0.0
        )
        if active_weight <= 0:
            raise ValueError("At least one reliability weight must be positive")
        if self.blur_reference <= 0:
            raise ValueError("blur_reference must be positive")
        if not 0 <= self.min_mask_area_ratio <= self.max_mask_area_ratio <= 1:
            raise ValueError("Mask area ratios must satisfy 0 <= min <= max <= 1")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def area_fraction(mask: np.ndarray) -> float:
    return float(mask.astype(bool).mean())


def area_change(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    previous_area = area_fraction(mask_b)
    current_area = area_fraction(mask_a)
    if previous_area == 0.0 and current_area == 0.0:
        return 0.0
    if previous_area == 0.0:
        return float("nan")
    return abs(current_area - previous_area)


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def centroid_shift(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    center_a = centroid(mask_a)
    center_b = centroid(mask_b)
    if center_a is None or center_b is None:
        return float("nan")
    return float(math.dist(center_a, center_b))

# implementation calculate grayscale Laplacian variance
def blur_score(frame_bgr: np.ndarray | None, blur_reference: float) -> float:
    if frame_bgr is None:
        return 1.0
    if cv2 is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        value = cv2.Laplacian(gray, cv2.CV_64F).var()
    else:
        gray = (
            frame_bgr.mean(axis=2).astype(np.float32)
            if frame_bgr.ndim == 3
            else frame_bgr.astype(np.float32)
        )
        laplacian = (
            -4.0 * gray
            + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
        )
        value = float(laplacian.var())
    return clamp01(float(value) / blur_reference)


def compute_reliability(
    current_mask: np.ndarray,
    previous_mask: np.ndarray | None,
    frame_bgr: np.ndarray | None,
    mask_confidence: float | None,
    config: ReliabilityConfig,
) -> dict[str, float]:
    """Compute a normalized reliability score and its component signals."""
    confidence_score = 0.5 if mask_confidence is None else clamp01(mask_confidence)
    if previous_mask is None or previous_mask.sum() == 0:
        temporal_score = 0.5
        area_score = 0.5
    else:
        temporal_score = mask_iou(current_mask, previous_mask)
        current_area = area_fraction(current_mask)
        previous_area = area_fraction(previous_mask)
        eps = config.area_epsilon
        area_score = math.exp(
            -abs(math.log((current_area + eps) / (previous_area + eps)))
        )


    blur_value = (
        blur_score(frame_bgr, config.blur_reference) if config.use_blur_score else 0.0
    )
    weighted_terms = [
        (config.confidence_weight, confidence_score),
        (config.temporal_weight, temporal_score),
        (config.area_weight, area_score),
    ]
    if config.use_blur_score:
        weighted_terms.append((config.blur_weight, blur_value))
    total_weight = sum(weight for weight, _ in weighted_terms)
    reliability = sum(weight * value for weight, value in weighted_terms) / total_weight

    current_area_ratio = area_fraction(current_mask)
    if (
        previous_mask is not None
        and previous_mask.sum() > 0
        and current_mask.sum() == 0
    ):
        reliability *= config.blank_mask_penalty
    if (
        current_area_ratio < config.min_mask_area_ratio
        or current_area_ratio > config.max_mask_area_ratio
    ):
        reliability *= config.implausible_area_penalty

    return {
        "reliability": clamp01(reliability),
        "r_conf": confidence_score,
        "r_temporal": temporal_score,
        "r_area": area_score,
        "r_blur": blur_value,
    }

# TODO: check this gate apply
# rejection counter prevents the output from remaining frozen indefinitely
def apply_reliability_gate(
    current_mask: np.ndarray,
    previous_valid_mask: np.ndarray | None,
    reliability: float,
    reliability_threshold: float,
    consecutive_rejections: int,
    max_consecutive_rejections: int,
) -> tuple[np.ndarray, bool, int]:
    """Select current versus prior output using the requested threshold."""
    if previous_valid_mask is None or reliability >= reliability_threshold:
        return current_mask.astype(np.uint8), True, 0
    consecutive_rejections += 1
    if consecutive_rejections > max_consecutive_rejections:
        return current_mask.astype(np.uint8), True, 0
    return previous_valid_mask.copy(), False, consecutive_rejections


def safe_nanmean(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))
