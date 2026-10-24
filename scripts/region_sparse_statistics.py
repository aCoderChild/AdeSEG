"""Offline bookkeeping composed with unchanged pinned author morphology."""

import ast
from functools import lru_cache

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from modeling.region_tokens import EFFICIENT_SAM2_REVISION, _boundary_iou, _verified_source


@lru_cache(maxsize=1)
def author_dilate():
    path, text = _verified_source(
        "Efficient-SAM2", EFFICIENT_SAM2_REVISION,
        "sam2/sam2_video_predictor_with_bypass_stg.py",
    )
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == "dilate_mask")
    namespace = {"F": F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def deficit_parts(pred, truth, ratio=0.02):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    if pred.shape != truth.shape or not truth.any():
        raise ValueError("Aligned GT-present binary masks required")
    tp = int((pred & truth).sum())
    fn, fp = truth & ~pred, pred & ~truth
    denominator = 2 * tp + int(fn.sum()) + int(fp.sum())
    parts = dict.fromkeys(("complete_miss", "edge", "undersegmentation", "excess_prediction"), 0.0)
    if not pred.any():
        parts["complete_miss"] = 1.0
    else:
        inner = _boundary_iou()(truth.astype(np.uint8), ratio).astype(bool)
        radius = max(1, int(round(ratio * np.hypot(*truth.shape))))
        # OpenCV's optimized rectangular dilation avoids a huge full-image
        # Torch pooling kernel; the square structuring element is explicit.
        band = cv2.dilate(inner.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)).astype(bool)
        parts["edge"] = float(((fn | fp) & band).sum() / denominator)
        parts["undersegmentation"] = float((fn & ~band).sum() / denominator)
        parts["excess_prediction"] = float((fp & ~band).sum() / denominator)
    dice = 2 * tp / denominator
    if not np.isclose(sum(parts.values()), 1 - dice, atol=1e-12):
        raise RuntimeError("Dice deficit partition does not sum exactly")
    return {"dice": dice, **parts, "zero_overlap_nonempty": bool(pred.any() and tp == 0)}


def token_counts(labels):
    labels = np.asarray(labels)
    if labels.shape != (64, 64) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Expected native 64x64 three-region labels")
    object_cells = labels != 0
    near = author_dilate()(torch.from_numpy(object_cells), 3).numpy() & ~object_cells
    counts = {"interior": int((labels == 1).sum()), "boundary": int((labels == 2).sum()),
              "near_background": int(near.sum()), "far_background": int(((labels == 0) & ~near).sum())}
    if sum(counts.values()) != 4096:
        raise RuntimeError("Four-way budget partition is not exhaustive")
    counts["retained"] = 4096 - counts["far_background"]
    counts["hypothetical_older_capacity"] = 20480 // counts["retained"] if counts["retained"] else None
    return counts


def pack_history(counts, budget=20480):
    """Counts of older frames, oldest first; preserve finite recent suffix."""
    if budget < 0 or any(not 0 <= c <= 4096 for c in counts):
        raise ValueError("Invalid spatial-key counts or budget")
    used = frames = 0
    for count in reversed(counts):
        if used + count > budget:
            break
        used += count
        frames += 1
    return {"older_frames": frames, "older_tokens": used}
