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
    boundary_reference: float = 30.0
    min_mask_area_ratio: float = 0.0005
    max_mask_area_ratio: float = 0.80
    implausible_area_penalty: float = 0.50
    object_absence_threshold: float = 0.5
    memory_prompt_min_peak: float = 0.05

    def __post_init__(self) -> None:
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
    clipped = np.clip(probs, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def memory_prompt_logits(
    memory_logits: np.ndarray,
    min_foreground_peak: float = 0.05,
) -> np.ndarray:
    """Zero out near-empty memory; otherwise reuse the stored logits as the prompt."""
    peak_prob = 1.0 / (1.0 + np.exp(-float(np.max(memory_logits))))
    if peak_prob < min_foreground_peak:
        return np.zeros_like(memory_logits, dtype=np.float32)
    return memory_logits.astype(np.float32)


def align_memory_to_frame(
    memory_logits: np.ndarray,
    previous_frame_bgr: np.ndarray | None,
    current_frame_bgr: np.ndarray,
) -> np.ndarray:
    """Warp memory forward from the previous frame to the current frame via optical flow."""
    if cv2 is None or previous_frame_bgr is None:
        return memory_logits.copy()

    height, width = memory_logits.shape
    previous_gray = cv2.cvtColor(previous_frame_bgr, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.resize(previous_gray, (width, height))
    current_gray = cv2.resize(current_gray, (width, height))

    # For each current-frame pixel, where did it come from in the previous frame.
    backward_flow = cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        memory_logits.astype(np.float32),
        grid_x + backward_flow[..., 0],
        grid_y + backward_flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,  # flowed-out-of-frame = neutral (logit 0 = p=0.5), not background
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


def boundary_score(
    frame_bgr: np.ndarray | None, mask: np.ndarray, boundary_reference: float
) -> float:
    """Does the mask edge coincide with a real image edge, or float in flat tissue?"""
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


def frame_specific_reliability(
    current_mask: np.ndarray,
    previous_mask: np.ndarray | None,
    frame_bgr: np.ndarray | None,
    mask_confidence: float,
    prompt_confidence: float | None,
    object_score: float,
    config: ReliabilityConfig,
    has_evidence: bool = True,
) -> dict[str, float]:
    """reliability from this frame's own measured signals."""
    confidence_score = clamp01(mask_confidence)
    object_score_value = clamp01(object_score)
    boundary_value = boundary_score(frame_bgr, current_mask, config.boundary_reference)

    scores = [confidence_score, boundary_value, object_score_value]
    prompt_confidence_score = None
    if prompt_confidence is not None:
        prompt_confidence_score = clamp01(prompt_confidence)
        scores.append(prompt_confidence_score)
    reliability = sum(scores) / len(scores)

    object_confidently_absent = object_score_value < config.object_absence_threshold
    current_area_ratio = area_fraction(current_mask)
    just_went_blank = (
        previous_mask is not None
        and previous_mask.sum() > 0
        and current_mask.sum() == 0
    )
    implausible_area = current_area_ratio > config.max_mask_area_ratio or (
        current_area_ratio < config.min_mask_area_ratio and not object_confidently_absent
    )
    no_evidence_hallucination = not has_evidence and current_area_ratio > 0

    if just_went_blank or no_evidence_hallucination:
        reliability = 0.0
    elif implausible_area:
        reliability *= config.implausible_area_penalty

    return {
        "reliability": clamp01(reliability),
        "r_conf": confidence_score,
        "r_prompt": prompt_confidence_score if prompt_confidence_score is not None else float("nan"),
        "r_boundary": boundary_value,
        "r_object": object_score_value,
    }


def select_reliability(
    signals: dict[str, float], fixed_value: float | None = None
) -> float:
    """(fixed_value set): same reliability every frame.
    (fixed_value None): signals["reliability"] from frame_specific_reliability."""
    if fixed_value is None:
        return signals["reliability"]
    return clamp01(fixed_value)


def pixelwise_reliability(current_probs: np.ndarray, reliability: float) -> np.ndarray:
    """Scale frame-level reliability by each pixel's own decisiveness."""
    decisiveness = np.abs(current_probs.astype(np.float32) - 0.5) * 2.0
    return clamp01(reliability) * decisiveness


def apply_reliability_gate(
    current_probs: np.ndarray,
    previous_memory_logits: np.ndarray | None,
    reliability: float,
) -> np.ndarray:
    """Fuse current + memory logits, weighted per pixel by reliability. Memory stays in logit space."""
    current_logit = logit(current_probs).astype(np.float32)
    if previous_memory_logits is None:
        return current_logit
    weight = pixelwise_reliability(current_probs, reliability)
    return weight * current_logit + (1.0 - weight) * previous_memory_logits.astype(np.float32)


def safe_nanmean(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))
