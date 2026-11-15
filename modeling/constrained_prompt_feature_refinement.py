"""Candidate-constrained variant of the existing feature-point prompt.

The existing project prototype and cosine contrast are unchanged. Restrict
positive candidates to the provisional mask and negative candidates to its
complement, both within the detector box. The mask is an automatic prediction,
not ground truth. This does not ensure semantic correctness.
"""
import numpy as np
import torch
import torch.nn.functional as F

from modeling.prompt_feature_refinement import reference_prototypes


def constrained_early_feature_points(backbone_out, provisional_mask, box):
    if not isinstance(backbone_out, dict) or not backbone_out.get('backbone_fpn'):
        raise ValueError('Missing Lean FPN outputs')
    features = backbone_out['backbone_fpn'][0]
    if not torch.is_tensor(features) or features.ndim != 4 or features.shape[0] != 1 or not torch.isfinite(features).all():
        raise ValueError('Expected finite single-image FPN output')
    mask = np.asarray(provisional_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Expected a 2D automatic mask')
    if not mask.any():
        return None, 'empty_provisional_mask'
    features = features[0].detach().float()
    try:
        positive, negative = reference_prototypes(features, mask)
    except ValueError:
        return None, 'degenerate_prototypes'
    height, width = mask.shape
    _, h, w = features.shape
    x0, y0, x1, y1 = np.asarray(box, dtype=float)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Invalid detector box')
    xx = (np.arange(w) + .5) * width / w
    yy = (np.arange(h) + .5) * height / h
    valid = (yy[:, None] >= y0) & (yy[:, None] < y1) & (xx[None, :] >= x0) & (xx[None, :] < x1)
    # Test the original-resolution binary mask at each exact feature-cell center.
    # No predicted confidence or probability threshold is introduced.
    yy_idx = np.minimum(yy.astype(int), height - 1)
    xx_idx = np.minimum(xx.astype(int), width - 1)
    candidate_mask = mask[yy_idx[:, None], xx_idx[None, :]]
    fg = np.flatnonzero(valid & candidate_mask)
    bg = np.flatnonzero(valid & ~candidate_mask)
    if len(fg) == 0 or len(bg) == 0:
        return None, 'insufficient_foreground_or_background_cells'
    normalized = F.normalize(features, dim=0)
    margin = (torch.einsum('chw,c->hw', normalized, positive) -
              torch.einsum('chw,c->hw', normalized, negative)).cpu().numpy().reshape(-1)
    positive_id = int(fg[np.argmax(margin[fg])])
    negative_id = int(bg[np.argmin(margin[bg])])
    def coordinate(index):
        return [float(xx[index % w]), float(yy[index // w])]
    return {'feature_points': [coordinate(positive_id), coordinate(negative_id)],
            'feature_margins': [float(margin[positive_id]), float(margin[negative_id])],
            'foreground_candidates': int(len(fg)), 'background_candidates': int(len(bg))}, 'available'
