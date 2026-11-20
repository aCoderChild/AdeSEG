"""Detector-conditioned provisional target anchors for SAM2 point initialization.

Unlike Lean-SAM2/TAMP's GT-initialized foreground anchors, these are pseudo-
anchors from a detector's box. They are not guaranteed to identify the lesion.
Always return a fixed positive/negative prompt budget or fail on invalid geometry.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def target_anchored_points(features, box, image_hw, *, positives=1, negatives=1,
                           max_anchors=4):
    """Select prompt points using diverse interior pseudo-anchors and background contrast.

    Args:
      features: frozen [C,Hf,Wf] or [1,C,Hf,Wf] early SAM2 FPN, image aligned.
      box: detector coordinates [x0, y0, x1, y1] in source-image pixels.
      image_hw: original source image [height, width].

    Uses no masks or future frames. Both prompt modalities are mandatory.
    """
    if not isinstance(features, torch.Tensor):
        raise ValueError("FGPI requires a frozen feature tensor")
    if len(image_hw) != 2 or any(type(x) is not int or x < 2 for x in image_hw):
        raise ValueError("Invalid original image geometry")
    if (type(positives) is not int or type(negatives) is not int
            or not (1 <= positives <= 8 and 1 <= negatives <= 8)
            or type(max_anchors) is not int or not 1 <= max_anchors <= 16):
        raise ValueError("Invalid joint point or anchor budget")
    height, width = image_hw
    if (len(box) != 4 or any(not math.isfinite(float(x)) for x in box)):
        raise ValueError("Invalid detector box")
    x0, y0, x1, y1 = map(float, box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Detector box outside original image")
    f = features.detach()
    if f.ndim == 4 and f.shape[0] == 1:
        f = f[0]
    if (f.ndim != 3 or f.shape[0] < 2 or min(f.shape[-2:]) < 4
            or not torch.is_floating_point(f) or not torch.isfinite(f).all()):
        raise ValueError("Expected finite floating early features [C,H,W]")
    f = f.float().cpu()
    _, fh, fw = f.shape
    y = (torch.arange(fh, dtype=torch.float32) + .5) * (height / fh)
    x = (torch.arange(fw, dtype=torch.float32) + .5) * (width / fw)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    xx, yy = xy[:,0], xy[:,1]
    bw, bh = x1-x0, y1-y0
    cx, cy = (x0+x1)/2, (y0+y1)/2
    inside = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    core = ((xx >= x0+.15*bw) & (xx <= x1-.15*bw)
            & (yy >= y0+.15*bh) & (yy <= y1-.15*bh))
    if not core.any():
        core = inside
    # Buffer outside the potentially inaccurate detector boundary.
    outside_buffer = ((xx < x0-.10*bw) | (xx > x1+.10*bw)
                      | (yy < y0-.10*bh) | (yy > y1+.10*bh))
    near = ((xx >= x0-.85*bw) & (xx <= x1+.85*bw)
            & (yy >= y0-.85*bh) & (yy <= y1+.85*bh))
    background = outside_buffer & near
    if int(core.sum()) < positives or int(background.sum()) < negatives:
        raise ValueError("Insufficient spatially separated foreground/background feature tokens")

    unit = F.normalize(f.flatten(1).T, p=2, dim=-1, eps=1e-6)
    # Conservative detector-interior prior; this defines *provisional* identity.
    radius = ((xx-cx)/(bw/2)).square() + ((yy-cy)/(bh/2)).square()
    central = torch.exp(-1.5*radius)
    core_indices = torch.where(core)[0]
    core_vectors = unit[core]
    center_weights = central[core]
    mean = F.normalize((core_vectors * center_weights[:,None]).sum(0), dim=0, eps=1e-6)
    # Restrict anchor diversity to the most spatially supported interior tokens.
    agreement = (core_vectors @ mean).clamp(-1,1)
    support = .80*agreement + .20*center_weights
    pool_size = min(core_indices.numel(), max(positives, max_anchors, int(math.ceil(.35*core_indices.numel()))))
    pool = sorted(range(len(core_indices)),key=lambda j:(-float(support[j]),int(core_indices[j])))[:pool_size]
    candidates = core_indices[pool]
    count = min(max_anchors, len(candidates))
    selected_anchors = [int(candidates[0])]
    while len(selected_anchors) < count:
        similarities = unit[candidates] @ unit[selected_anchors].T
        max_sim = similarities.max(dim=1).values
        chosen = set(selected_anchors)
        options = [j for j,i in enumerate(candidates.tolist()) if i not in chosen]
        # Lean-SAM2-inspired diverse token selection, but with center support
        # to avoid arbitrary distant/distractor patches in the YOLO box.
        best = min(options,key=lambda j:(-float(.7*(1-max_sim[j])+.3*central[candidates[j]]),int(candidates[j])))
        selected_anchors.append(int(candidates[best]))
    anchors = unit[selected_anchors]
    # Max-cosine pseudo-target identity; distinct from true prompt mask anchors.
    fg = (unit @ anchors.T).max(1).values
    bg_proto = F.normalize(unit[background].mean(0), dim=0, eps=1e-6)
    bg = unit @ bg_proto
    contrast = (fg-bg).reshape(1,1,fh,fw)
    smoothed = F.avg_pool2d(contrast, kernel_size=3,stride=1,padding=1).reshape(-1)
    score = .65*contrast.reshape(-1) + .35*smoothed
    pos_score = score + .12*central
    # Prefer relevant nearby outside-background over unrelated image corners.
    dx = (x0-xx).clamp_min(0)+(xx-x1).clamp_min(0)
    dy = (y0-yy).clamp_min(0)+(yy-y1).clamp_min(0)
    exterior_distance = torch.sqrt(dx.square()+dy.square())
    near_prior = 1/(1+exterior_distance/(.35*min(bw,bh)))
    neg_score = -score + .25*near_prior

    def select(mask, values, number):
        ids = sorted(torch.where(mask)[0].tolist(),key=lambda i:(-float(values[i]),i))
        chosen = []
        distance = .22*min(bw,bh)
        for index in ids:
            if all(float(torch.norm(xy[index]-xy[j])) >= distance for j in chosen):
                chosen.append(index)
            if len(chosen) == number:
                break
        # Tight boxes can have too few spatially separated tokens. Still use
        # distinct best-scoring positions rather than silently dropping points.
        if len(chosen) < number:
            for index in ids:
                if index not in chosen:
                    chosen.append(index)
                if len(chosen) == number:
                    break
        if len(chosen) != number:
            raise ValueError("Unable to satisfy fixed feature-point budget")
        return chosen

    pids = select(core,pos_score,positives)
    nids = select(background,neg_score,negatives)
    ids = pids + nids
    return {
        "point_coords": [[float(x) for x in xy[i]] for i in ids],
        "point_labels": [1]*positives + [0]*negatives,
        "anchor_coords": [[float(x) for x in xy[i]] for i in selected_anchors],
        "anchor_count": len(selected_anchors),
        "feature_contrast": float(score[pids].mean()-score[nids].mean()),
        "selection_strategy": "box_pseudo_anchor_cosine_bg_contrast_spatial_v1",
    }
