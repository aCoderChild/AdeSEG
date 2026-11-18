"""Select automatic box-refinement points from Lean's early FPN features."""
import numpy as np
import torch

from modeling.prompt_feature_refinement import reference_prototypes, select_points


def early_feature_points(backbone_out, provisional_mask, box, seed):
    """Use an automatic preliminary mask; return None for documented fallbacks."""
    if not isinstance(backbone_out, dict) or not backbone_out.get('backbone_fpn'):
        raise ValueError('Missing Lean FPN outputs')
    feature = backbone_out['backbone_fpn'][0]
    if not torch.is_tensor(feature) or feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError('Expected a single early FPN feature map')
    if not torch.isfinite(feature).all():
        raise ValueError('Nonfinite early FPN features')
    mask = np.asarray(provisional_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Expected a 2D automatic mask')
    if not mask.any():
        return None, 'empty_provisional_mask'
    try:
        foreground, background = reference_prototypes(feature[0].detach().float(), mask)
    except ValueError:
        return None, 'degenerate_prototypes'
    points = select_points(feature[0].detach().float(), foreground, background, box, mask.shape, seed)
    if points is None:
        return None, 'insufficient_box_cells'
    return points, 'available'
