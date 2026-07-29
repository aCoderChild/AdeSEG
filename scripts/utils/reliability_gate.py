"""Reliability scoring and temporal metrics for video mask experiments."""

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
    prompt_confidence_weight: float = 0.25
    boundary_weight: float = 0.30
    blur_weight: float = 0.10
    object_score_weight: float = 0.20
    use_blur_score: bool = True
    blur_reference: float = 150.0
    boundary_reference: float = 30.0
    min_mask_area_ratio: float = 0.0005
    max_mask_area_ratio: float = 0.80
    blank_mask_penalty: float = 0.25  # penalties
    implausible_area_penalty: float = 0.50  # penalties
    no_evidence_penalty: float = 0.15
    object_absence_threshold: float = 0.5
    memory_prompt_min_peak: float = 0.05

    # hard coded weights
    def __post_init__(self) -> None:
        weights = (
            self.confidence_weight,
            self.prompt_confidence_weight,
            self.boundary_weight,
            self.blur_weight,
            self.object_score_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Reliability weights must be non-negative")
        active_weight = (
            self.confidence_weight
            + self.prompt_confidence_weight
            + self.boundary_weight
            + self.object_score_weight
            + (self.blur_weight if self.use_blur_score else 0.0)
        )
        if active_weight <= 0:
            raise ValueError("At least one reliability weight must be positive")
        if self.blur_reference <= 0:
            raise ValueError("blur_reference must be positive")
        if self.boundary_reference <= 0:
            raise ValueError("boundary_reference must be positive")
        if not 0 <= self.min_mask_area_ratio <= self.max_mask_area_ratio <= 1:
            raise ValueError("Mask area ratios must satisfy 0 <= min <= max <= 1")
        if not 0 <= self.object_absence_threshold <= 1:
            raise ValueError("object_absence_threshold must be in [0, 1]")
        if not 0 <= self.memory_prompt_min_peak <= 1:
            raise ValueError("memory_prompt_min_peak must be in [0, 1]")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def logit(probs: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Convert probabilities to finite logits for MedSAM2 mask prompting."""
    clipped = np.clip(probs, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))

# TODO
def memory_prompt_logits(
    memory_probs: np.ndarray,
    min_foreground_peak: float = 0.05,
) -> np.ndarray:
    """Convert fused memory to a scale-stable dense prompt.

    Reliability fusion can lower the whole probability map even when its
    spatial shape is correct. Peak normalization preserves that shape for the
    prompt encoder without changing the stored memory or its update equation.
    Near-empty memories stay neutral instead of amplifying numerical noise.
    """
    peak = float(np.max(memory_probs))
    if peak < min_foreground_peak:
        return np.zeros_like(memory_probs, dtype=np.float32)
    normalized = np.clip(memory_probs / peak, 0.0, 1.0)
    return logit(normalized).astype(np.float32)

# a frame has no fresh box prompt
# script still has to reuse the memory mask from previous frame
# camera/tissue moved between frames => memory could MISALIGN
def align_memory_to_frame(
    memory_probs: np.ndarray,
    previous_frame_bgr: np.ndarray | None,
    current_frame_bgr: np.ndarray,
) -> np.ndarray:
    """shift memory mask to follow camera's motion. 
    Ex: previous frame has the object but the current frame NOT"""
    if cv2 is None or previous_frame_bgr is None: # first frame
        return memory_probs.copy()

    height, width = memory_probs.shape
    previous_gray = cv2.cvtColor(previous_frame_bgr, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.resize(previous_gray, (width, height))
    current_gray = cv2.resize(current_gray, (width, height))

    # Backward flow maps each current-frame pixel to its source in the
    # preceding frame, which is the coordinate convention cv2.remap needs.
    backward_flow = cv2.calcOpticalFlowFarneback( # compute flow from first arg to second
        current_gray,
        previous_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    # warp the memory according to the shift
    return cv2.remap(
        memory_probs.astype(np.float32),
        grid_x + backward_flow[..., 0],
        grid_y + backward_flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0, # "no evidence"
    )


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
    object_score: float | None = None,
    has_evidence: bool = True,
) -> dict[str, float]:
    confidence_score = 0.5 if mask_confidence is None else clamp01(mask_confidence)
    prompt_confidence_score = (
        0.5 if prompt_confidence is None else clamp01(prompt_confidence)
    )
    object_score_value = 0.5 if object_score is None else clamp01(object_score)

    boundary_value = boundary_score(frame_bgr, current_mask, config.boundary_reference)
    blur_value = (
        blur_score(frame_bgr, config.blur_reference) if config.use_blur_score else 0.0
    )
    weighted_terms = [
        (config.confidence_weight, confidence_score),
        (config.prompt_confidence_weight, prompt_confidence_score),
        (config.boundary_weight, boundary_value),
        (config.object_score_weight, object_score_value),
    ]
    if config.use_blur_score:
        weighted_terms.append((config.blur_weight, blur_value))
    total_weight = sum(weight for weight, _ in weighted_terms)
    reliability = sum(weight * value for weight, value in weighted_terms) / total_weight

    object_confidently_absent = (
        object_score is not None and object_score_value < config.object_absence_threshold
    )

    current_area_ratio = area_fraction(current_mask)
    just_went_blank = (
        previous_mask is not None
        and previous_mask.sum() > 0
        and current_mask.sum() == 0
    )
    too_small = current_area_ratio < config.min_mask_area_ratio
    too_large = current_area_ratio > config.max_mask_area_ratio

    no_evidence_hallucination = not has_evidence and current_area_ratio > 0
    # Do not let one low object-presence score immediately erase an object
    # that was visible in the preceding frame.
    if just_went_blank:
        reliability *= config.blank_mask_penalty
    if too_large or (too_small and not object_confidently_absent):
        reliability *= config.implausible_area_penalty
    if no_evidence_hallucination:
        reliability *= config.no_evidence_penalty

    return {
        "reliability": clamp01(reliability),
        "r_conf": confidence_score,
        "r_prompt": prompt_confidence_score,
        "r_boundary": boundary_value,
        "r_blur": blur_value,
        "r_object": object_score_value,
        "no_evidence_penalty_applied": no_evidence_hallucination,
    }


def select_reliability(
    signals: dict[str, float], fixed_reliability: float | None = None
) -> float:
    """Return the heuristic score, or a fixed value for an ablation."""
    if fixed_reliability is None:
        return signals["reliability"]
    return clamp01(fixed_reliability)


def apply_reliability_gate(
    current_mask: np.ndarray,
    previous_memory_mask: np.ndarray | None,
    reliability: float,
) -> np.ndarray:
    """Fuse the current probability mask into the single memory every frame."""
    current = current_mask.astype(np.float32)
    if previous_memory_mask is None:
        return current.copy()
    previous = previous_memory_mask.astype(np.float32)
    weight = clamp01(reliability)
    return weight * current + (1.0 - weight) * previous


def safe_nanmean(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))
