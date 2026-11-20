"""FGPI: joint YOLO detection box and target-anchored early-feature prompts.

The box_only mode is an independent baseline, NOT a reliability fallback.
The FGPI branch always adds positive/negative points or raises for invalid
spatial geometry. A detector box supplies provisional identity, not trusted GT.
"""
from __future__ import annotations

import math
import torch

from adeseg.prompts.target_anchored import target_anchored_points


def _validate_box(box_xyxy, image_hw):
    if len(image_hw) != 2 or any(type(v) is not int or v < 2 for v in image_hw):
        raise ValueError("image_hw must be two positive integer pixel dimensions")
    height, width = image_hw
    if not isinstance(box_xyxy, (list, tuple)) or len(box_xyxy) != 4:
        raise ValueError("Detector box must be [x0,y0,x1,y1]")
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in box_xyxy):
        raise ValueError("Detector box must contain finite coordinates")
    x0, y0, x1, y1 = (float(x) for x in box_xyxy)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Detector box is outside the image or has zero area")
    return [x0, y0, x1, y1]



@torch.no_grad()
def feature_guided_prompt(features, box_xyxy, image_hw, *,
                          detector_confidence=1.0, min_detector_confidence=0.0,
                          mode="fgpi", positives=1, negatives=1,
                          max_anchors=4):
    """Joint detector-box + pseudo-target-anchor points without confidence gating.

    The default detector admission threshold is zero. Nonzero admission
    thresholds, when specified, apply equally to both experimental arms;
    they never conditionally remove FGPI feature-derived points.
    """
    box = _validate_box(box_xyxy, image_hw)
    if mode not in ("box_only", "fgpi"):
        raise ValueError("FGPI mode must be box_only or fgpi")
    if (type(detector_confidence) not in (float, int)
            or type(min_detector_confidence) not in (float, int)
            or not 0 <= detector_confidence <= 1
            or not 0 <= min_detector_confidence <= 1):
        raise ValueError("Invalid detector confidence or gate")
    if detector_confidence < min_detector_confidence:
        raise ValueError("Detector prediction below predeclared confidence threshold")
    result = {
        "box_xyxy": box, "point_coords": [], "point_labels": [],
        "mode": mode, "decision": "box_only", "reason": "control",
        "detector_confidence": float(detector_confidence),
        "feature_contrast": None,
    }
    if mode == "box_only":
        return result
    points = target_anchored_points(
        features, box, image_hw, positives=positives,
        negatives=negatives, max_anchors=max_anchors
    )
    result.update(points)
    result["decision"] = "augmented"
    result["reason"] = "mandatory_target_anchored_feature_points"
    return result


def apply_fgpi_prompt(predictor, state, frame_index, obj_id, prompt):
    """Call the VENDORED SAM2 box-plus-points interface without re-prompting GT."""
    if not isinstance(prompt, dict) or not {"box_xyxy","point_coords","point_labels"}.issubset(prompt):
        raise ValueError("Invalid FGPI prompt")
    if len(prompt["point_coords"]) != len(prompt["point_labels"]):
        raise ValueError("FGPI points and labels are misaligned")
    if any(x not in (0, 1) for x in prompt["point_labels"]):
        raise ValueError("FGPI points must have foreground/background labels")
    if prompt.get("mode") == "fgpi":
        if (prompt.get("decision") != "augmented"
                or 1 not in prompt["point_labels"] or 0 not in prompt["point_labels"]):
            raise ValueError("FGPI must always include feature-guided foreground and background points")
    elif prompt.get("mode") == "box_only":
        if prompt["point_coords"] or prompt["point_labels"]:
            raise ValueError("Box-only control must have no extra points")
    else:
        raise ValueError("Unsupported FGPI prompt mode")
    return predictor.add_new_points_or_box(
        state, frame_idx=frame_index, obj_id=obj_id,
        box=prompt["box_xyxy"],
        points=prompt["point_coords"] or None,
        labels=prompt["point_labels"] or None,
        clear_old_points=True, normalize_coords=True,
    )
