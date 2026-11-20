"""FGPI: detector-box-conditioned, training-free early-feature point proposals.

Uses ONLY frozen image-encoder features and an automatic detector's box. No
ground truth, predicted masks, teacher logits or future frames enter prompts.
This is an *experimental* box-plus-points policy; no clinical benefit is claimed.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


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


def _choose(scores, eligibility, coordinates, n, min_distance):
    """Stable highest-score selection with spatial suppression."""
    candidates = torch.where(eligibility)[0].tolist()
    candidates.sort(key=lambda i: (-float(scores[i]), i))
    chosen = []
    for index in candidates:
        xy = coordinates[index]
        if all(float(torch.sum((xy - coordinates[old]) ** 2)) >= min_distance ** 2
               for old in chosen):
            chosen.append(index)
        if len(chosen) >= n:
            break
    return chosen


@torch.no_grad()
def feature_guided_prompt(features, box_xyxy, image_hw, *,
                          detector_confidence=1.0, min_detector_confidence=0.25,
                          mode="fgpi", positives=1, negatives=1,
                          min_feature_contrast=0.1):
    """Return one SAM2-compatible box, optionally with image-coordinate points.

    FG/BG prototypes are computed from the inner detector box and an outside
    annulus. Per-pixel cosine contrast ranks candidate foreground/background
    points; a weak or invalid contrast falls back to the unchanged YOLO box.
    Feature tensor must be [C,Hf,Wf] or [1,C,Hf,Wf], spatially aligned with
    original image coordinates and extracted without consulting ground truth.
    """
    box = _validate_box(box_xyxy, image_hw)
    if mode not in ("box_only", "fgpi"):
        raise ValueError("FGPI mode must be box_only or fgpi")
    if (not 0 <= detector_confidence <= 1 or
            not 0 <= min_detector_confidence <= 1):
        raise ValueError("Invalid detector confidence or gate")
    if type(positives) is not int or type(negatives) is not int or not (
            1 <= positives <= 8 and 0 <= negatives <= 8):
        raise ValueError("Unsupported FGPI point budget")
    if not 0 <= min_feature_contrast <= 2:
        raise ValueError("Invalid feature contrast threshold")
    result = {
        "box_xyxy": box, "point_coords": [], "point_labels": [],
        "mode": mode, "decision": "box_only", "reason": "control",
        "detector_confidence": float(detector_confidence),
        "feature_contrast": None,
    }
    if detector_confidence < min_detector_confidence:
        raise ValueError("Detector prediction below predeclared confidence threshold")
    if mode == "box_only":
        return result
    if not isinstance(features, torch.Tensor):
        raise ValueError("FGPI requires an actual frozen image-encoder feature tensor")
    f = features.detach()
    if f.ndim == 4 and f.shape[0] == 1:
        f = f[0]
    if (f.ndim != 3 or f.shape[0] < 2 or min(f.shape[-2:]) < 4 or
            not torch.is_floating_point(f) or not torch.isfinite(f).all()):
        raise ValueError("Early features must be finite float [C,Hf,Wf]")
    # CPU computation avoids device-dependent unstable equal-score ordering.
    f = f.float().cpu()
    _, fh, fw = f.shape
    height, width = image_hw
    yy, xx = torch.meshgrid((torch.arange(fh, dtype=torch.float32) + 0.5) *
                             (height / fh), (torch.arange(fw, dtype=torch.float32) + 0.5) *
                             (width / fw), indexing="ij")
    coords = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    x0, y0, x1, y1 = box
    bw, bh = x1-x0, y1-y0
    inside = ((coords[:, 0] >= x0) & (coords[:, 0] <= x1)
              & (coords[:, 1] >= y0) & (coords[:, 1] <= y1))
    core = ((coords[:, 0] >= x0 + 0.15*bw) & (coords[:, 0] <= x1 - 0.15*bw)
            & (coords[:, 1] >= y0 + 0.15*bh)
            & (coords[:, 1] <= y1 - 0.15*bh))
    if not core.any():
        core = inside
    ring = ((coords[:, 0] >= x0 - 0.5*bw) & (coords[:, 0] <= x1 + 0.5*bw)
            & (coords[:, 1] >= y0 - 0.5*bh)
            & (coords[:, 1] <= y1 + 0.5*bh) & ~inside)
    if not core.any() or not ring.any():
        result["reason"] = "insufficient_region_context"
        return result

    unit = F.normalize(f.flatten(1).T, dim=1, eps=1e-6)
    fg = F.normalize(unit[core].mean(dim=0), dim=0, eps=1e-6)
    bg = F.normalize(unit[ring].mean(dim=0), dim=0, eps=1e-6)
    contrast = (unit @ fg) - (unit @ bg)
    separation = float(contrast[core].max() - contrast[ring].min())
    result["feature_contrast"] = separation
    if not math.isfinite(separation) or separation < min_feature_contrast:
        result["reason"] = "weak_feature_discrimination"
        return result

    distance = max(2.0, 0.08 * min(bw, bh))
    foreground = _choose(contrast, core, coords, positives, distance)
    background = _choose(-contrast, ring, coords, negatives, distance)
    if not foreground or (negatives > 0 and not background):
        result["reason"] = "insufficient_point_candidates"
        return result
    indices = foreground + background
    result.update({
        "point_coords": [[float(v) for v in coords[i]] for i in indices],
        "point_labels": [1] * len(foreground) + [0] * len(background),
        "decision": "augmented", "reason": "foreground_background_feature_contrast",
    })
    return result


def apply_fgpi_prompt(predictor, state, frame_index, obj_id, prompt):
    """Call the VENDORED SAM2 box-plus-points interface without re-prompting GT."""
    if not isinstance(prompt, dict) or not {"box_xyxy","point_coords","point_labels"}.issubset(prompt):
        raise ValueError("Invalid FGPI prompt")
    if len(prompt["point_coords"]) != len(prompt["point_labels"]):
        raise ValueError("FGPI points and labels are misaligned")
    if any(x not in (0, 1) for x in prompt["point_labels"]):
        raise ValueError("FGPI points must have foreground/background labels")
    return predictor.add_new_points_or_box(
        state, frame_idx=frame_index, obj_id=obj_id,
        box=prompt["box_xyxy"],
        points=prompt["point_coords"] or None,
        labels=prompt["point_labels"] or None,
        clear_old_points=True, normalize_coords=True,
    )
