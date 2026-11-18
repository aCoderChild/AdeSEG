"""Training-free target/context point prompts from native SAM2 image features.

This is a project diagnostic, not the CMSA or TCMN algorithm. The reference
mask must be an automatic prediction; target-frame GT is never accepted.
"""
import numpy as np
import torch
import torch.nn.functional as F


def reference_prototypes(features, initial_mask):
    """Weighted prompt foreground and safely separated background prototypes."""
    if features.ndim != 3 or not torch.isfinite(features).all():
        raise ValueError('Expected finite [C,H,W] features')
    mask = torch.as_tensor(initial_mask, dtype=torch.float32, device=features.device)
    if mask.ndim != 2 or not mask.any():
        raise ValueError('Initial prompt must contain foreground')
    size = features.shape[-2:]
    weights = F.interpolate(mask[None, None], size=size, mode='area')[0, 0]
    # Exclude a two-cell margin around any foreground coverage from context.
    blocked = F.max_pool2d((weights > 0).float()[None, None], 5, stride=1, padding=2)[0, 0]
    background = (blocked == 0).float()
    if weights.sum() <= 0 or background.sum() == 0:
        raise ValueError('Reference lacks foreground or uncontaminated background')
    vectors = features.float()
    positive = (vectors * weights).sum((1, 2)) / weights.sum()
    negative = (vectors * background).sum((1, 2)) / background.sum()
    if positive.norm() == 0 or negative.norm() == 0:
        raise ValueError('Degenerate reference prototype')
    return F.normalize(positive, dim=0), F.normalize(negative, dim=0)


def select_points(features, positive, negative, box_xyxy, image_hw, seed):
    """Feature and random-control point pairs within the same detector box.

    Cell centers are mapped directly to original pixel coordinates because
    SAM2ImagePredictor stretches its input to a square before image encoding.
    Returns None when the box covers fewer than two feature-cell centers.
    """
    if features.ndim != 3 or not torch.isfinite(features).all():
        raise ValueError('Expected finite [C,H,W] features')
    c, h, w = features.shape
    if positive.shape != (c,) or negative.shape != (c,):
        raise ValueError('Prototype channels do not match')
    height, width = image_hw
    x0, y0, x1, y1 = np.asarray(box_xyxy, dtype=float)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Invalid detector box')
    xx = (np.arange(w) + .5) * width / w
    yy = (np.arange(h) + .5) * height / h
    valid = (yy[:, None] >= y0) & (yy[:, None] < y1) & (xx[None, :] >= x0) & (xx[None, :] < x1)
    positions = np.flatnonzero(valid)
    if len(positions) < 2:
        return None
    normalized = F.normalize(features.float(), dim=0)
    margin = (torch.einsum('chw,c->hw', normalized, positive.float()) -
              torch.einsum('chw,c->hw', normalized, negative.float())).cpu().numpy().reshape(-1)
    order = positions[np.argsort(margin[positions], kind='stable')]
    feature_ids = [int(order[-1]), int(order[0])]
    random_ids = np.random.default_rng(seed).choice(positions, size=2, replace=False).tolist()
    def coordinates(ids):
        return [[float(xx[i % w]), float(yy[i // w])] for i in ids]
    return {'feature_points': coordinates(feature_ids), 'random_points': coordinates(random_ids),
            'feature_margins': [float(margin[i]) for i in feature_ids],
            'candidate_cells': int(len(positions))}
