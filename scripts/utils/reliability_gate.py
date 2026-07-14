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
    # confidence_weight: SAM2's own per-frame object-score confidence
    # (external/MedSAM2/medsam2_infer_video_with_yolo.py persists
    # sigmoid(object_score_logits) to a confidence CSV per sequence).
    # prompt_confidence_weight: confidence of the box that *seeded* this
    # frame (YOLO detection score, or 1.0 for a ground-truth box), persisted
    # to a separate prompt_confidence CSV. This is independent of
    # confidence_weight -- one reflects how trustworthy the input prompt
    # was, the other how confident the decoder is in its output -- and is
    # 0.5 (neutral) on frames with no prompt at all, e.g. stride>1 gaps
    # with the memory bank disabled.
    # No area_weight: comparing the candidate's size against the previous
    # memory mask is self-referential (previous_mask here is our own
    # blended output), the same issue that ruled out a temporal term.
    confidence_weight: float = 0.35
    prompt_confidence_weight: float = 0.25
    boundary_weight: float = 0.30
    blur_weight: float = 0.10
    use_blur_score: bool = True
    blur_reference: float = 150.0
    # boundary_reference normalizes mean Sobel-gradient magnitude sampled
    # along the mask edge; like blur_reference, it's a heuristic starting
    # point rather than a value calibrated against this dataset.
    boundary_reference: float = 30.0
    min_mask_area_ratio: float = 0.0005
    max_mask_area_ratio: float = 0.80
    blank_mask_penalty: float = 0.25  # penalties
    implausible_area_penalty: float = 0.50  # penalties

    # hard coded weights
    def __post_init__(self) -> None:
        weights = (
            self.confidence_weight,
            self.prompt_confidence_weight,
            self.boundary_weight,
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
        if self.boundary_reference <= 0:
            raise ValueError("boundary_reference must be positive")
        if not 0 <= self.min_mask_area_ratio <= self.max_mask_area_ratio <= 1:
            raise ValueError("Mask area ratios must satisfy 0 <= min <= max <= 1")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


# grayscale Laplacian variance, normalized by blur_reference
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
        # Reflect-pad to match cv2's default border handling; np.roll would
        # wrap pixels across opposite edges and contaminate the variance.
        padded = np.pad(gray, 1, mode="reflect")
        laplacian = (
            -4.0 * gray
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        value = float(laplacian.var())
    return clamp01(float(value) / blur_reference)


def _dilate_once(mask_bool: np.ndarray) -> np.ndarray:
    padded = np.pad(mask_bool, 1, mode="constant", constant_values=False)
    return (
        padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
        | padded[1:-1, 1:-1]
    )


def _erode_once(mask_bool: np.ndarray) -> np.ndarray:
    padded = np.pad(mask_bool, 1, mode="constant", constant_values=True)
    return (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[1:-1, 1:-1]
    )


def mask_boundary_ring(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """A thin ring straddling the mask edge, used to sample image gradient."""
    mask_bool = mask.astype(bool)
    dilated = mask_bool
    eroded = mask_bool
    for _ in range(iterations):
        dilated = _dilate_once(dilated)
        eroded = _erode_once(eroded)
    return dilated & ~eroded


# Does the mask edge coincide with a real intensity/color edge in the frame,
# or is it floating in a visually uniform region (a sign of a spurious mask
# that is shape-plausible and temporally stable but not anchored to any
# actual tissue boundary)?
def boundary_score(
    frame_bgr: np.ndarray | None, mask: np.ndarray, boundary_reference: float
) -> float:
    if frame_bgr is None or mask.sum() == 0:
        return 0.5
    if cv2 is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    else:
        gray = (
            frame_bgr.mean(axis=2).astype(np.float32)
            if frame_bgr.ndim == 3
            else frame_bgr.astype(np.float32)
        )
        padded = np.pad(gray, 1, mode="reflect")
        grad_x = padded[1:-1, 2:] - padded[1:-1, :-2]
        grad_y = padded[2:, 1:-1] - padded[:-2, 1:-1]
    gradient = np.hypot(grad_x, grad_y)

    ring = mask_boundary_ring(mask)
    if ring.sum() == 0:
        return 0.5
    value = float(gradient[ring].mean())
    return clamp01(value / boundary_reference)


def compute_reliability(
    current_mask: np.ndarray,
    previous_mask: np.ndarray | None,
    frame_bgr: np.ndarray | None,
    mask_confidence: float | None,
    prompt_confidence: float | None,
    config: ReliabilityConfig,
) -> dict[str, float]:
    """Compute a normalized reliability score and its component signals.

    Features: SAM2's own decoder confidence, the confidence of the prompt
    that seeded this frame (YOLO detection score, or 1.0 for a ground-truth
    box; neutral 0.5 on frames with no prompt), boundary alignment (does the
    mask edge sit on a real intensity edge in the frame -- the signal here
    that reads actual image content instead of only mask/model self-
    reports), and optional blur. There is no area-ratio term: with the
    memory bank disabled, each candidate is already an independent
    per-frame prediction, and `previous_mask` here is our own blended
    output, so comparing size against it would partly measure
    self-agreement rather than an independent signal. `previous_mask` is
    still used below for the blank-mask penalty.
    """
    confidence_score = 0.5 if mask_confidence is None else clamp01(mask_confidence)
    prompt_confidence_score = (
        0.5 if prompt_confidence is None else clamp01(prompt_confidence)
    )

    boundary_value = boundary_score(frame_bgr, current_mask, config.boundary_reference)
    blur_value = (
        blur_score(frame_bgr, config.blur_reference) if config.use_blur_score else 0.0
    )
    weighted_terms = [
        (config.confidence_weight, confidence_score),
        (config.prompt_confidence_weight, prompt_confidence_score),
        (config.boundary_weight, boundary_value),
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
        "r_prompt": prompt_confidence_score,
        "r_boundary": boundary_value,
        "r_blur": blur_value,
    }


def apply_reliability_gate(
    current_mask: np.ndarray,
    previous_memory_mask: np.ndarray | None,
    reliability: float,
) -> np.ndarray:
    """Soft memory update: M_t = r_t * Z_t + (1 - r_t) * M_(t-1).

    Z_t is the current frame's candidate mask, M_(t-1) is the previous
    memory mask (a float array in [0, 1]), and r_t is `reliability`. Every
    frame updates memory -- there is no accept/reject decision or rejection
    counter; `reliability` alone controls how much the new candidate
    contributes versus how much of the prior memory is retained. On the
    first frame (no memory yet), M_t = Z_t.
    """
    current = current_mask.astype(np.float32)
    if previous_memory_mask is None:
        return current
    r = clamp01(reliability)
    return r * current + (1.0 - r) * previous_memory_mask.astype(np.float32)


def safe_nanmean(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))
