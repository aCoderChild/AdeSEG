"""Blend an initial-prompt FPN contrast into Lean's existing anchor score.

The author model still determines usage, the fixed token budget, temporal
memory, routing, and output masks. This experimental adapter only changes the
spatial score consumed by the existing top-k selection; it does not alter
TARR's already-queued routing affinity.
"""
from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F

from modeling.prompt_feature_refinement import reference_prototypes


def blend_anchor_with_early_contrast(anchor, feature, positive, negative, spatial_size, strength):
    """Fixed convex blend of Lean anchor cosine and projected FPN contrast."""
    if not 0 <= strength <= 1:
        raise ValueError('Blend strength must lie in [0, 1]')
    if feature.ndim != 3 or positive.shape != (feature.shape[0],) or negative.shape != positive.shape:
        raise ValueError('Feature and prototype shapes differ')
    height, width = spatial_size
    base = anchor.detach().float().reshape(-1)
    if base.numel() != height * width or not torch.isfinite(base).all() or not torch.isfinite(feature).all():
        raise ValueError('Invalid anchor or feature')
    if strength == 0:
        return base
    normalized = F.normalize(feature.detach().float(), dim=0)
    positive_map = torch.einsum('chw,c->hw', normalized, positive.float())
    negative_map = torch.einsum('chw,c->hw', normalized, negative.float())
    # A difference of two cosines lies in [-2, 2]; rescale to the existing
    # anchor cosine's [-1, 1] range, then average to the memory-token grid.
    contrast = (positive_map - negative_map)[None, None] * 0.5
    contrast = F.interpolate(contrast, size=(height, width), mode='area').reshape(-1)
    result = (1 - strength) * base.to(contrast.device) + strength * contrast
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite blended score')
    return result


@contextmanager
def early_contrast_anchor(model, strength, observations):
    """Install after the existing pooled-anchor wrapper, for one isolated run."""
    if not 0 <= strength <= 1:
        raise ValueError('Blend strength must lie in [0, 1]')
    get_feature = model._get_image_feature
    add_mask = model.add_new_mask
    cache_score = model._cache_tarr_affinity_from_new_memory
    latest = {'frame': None, 'feature': None, 'prototypes': None}

    def observed_feature(self, inference_state, frame_idx, batch_size):
        result = get_feature(inference_state, frame_idx, batch_size)
        if batch_size != 1:
            raise ValueError('One object is required')
        feature = result[1]['backbone_fpn'][0]
        if feature.ndim != 4 or feature.shape[0] != 1:
            raise ValueError('Unexpected early FPN feature')
        latest['frame'] = int(frame_idx)
        latest['feature'] = feature[0].detach().float()
        return result

    def observed_mask(self, inference_state, frame_idx, obj_id, mask):
        if latest['frame'] != int(frame_idx) or latest['feature'] is None:
            raise ValueError('Initial mask and early feature frame differ')
        if latest['prototypes'] is not None:
            raise ValueError('Only one initial mask is supported')
        latest['prototypes'] = reference_prototypes(latest['feature'], mask)
        return add_mask(inference_state, frame_idx=frame_idx, obj_id=obj_id, mask=mask)

    def blended_cache(self, current_out, output_dict, frame_idx, obj_idx, is_init_cond_frame, run_mem_encoder, reverse):
        cache_score(current_out, output_dict, frame_idx, obj_idx, is_init_cond_frame, run_mem_encoder, reverse)
        base = current_out.get('anchor_similarity_cache') if isinstance(current_out, dict) else None
        if base is None:
            return  # The author has not produced an anchor score for this frame.
        if latest['prototypes'] is None or latest['feature'] is None or latest['frame'] != int(frame_idx):
            raise ValueError('Early contrast source is missing or from a different frame')
        memory = current_out.get('maskmem_features')
        if not torch.is_tensor(memory) or memory.ndim != 4 or memory.shape[0] != 1:
            raise ValueError('Unexpected memory feature grid')
        positive, negative = latest['prototypes']
        with torch.no_grad():
            changed = blend_anchor_with_early_contrast(base, latest['feature'], positive, negative, memory.shape[-2:], strength)
        if strength:
            current_out['anchor_similarity_cache'] = changed.to(device=base.device, dtype=base.dtype)
        observations.append({'frame_index': int(frame_idx), 'object_index': int(obj_idx),
                             'tokens': int(base.numel()),
                             'mean_abs_score_change': float((changed - base.float()).abs().mean().item())})

    model._get_image_feature = MethodType(observed_feature, model)
    model.add_new_mask = MethodType(observed_mask, model)
    model._cache_tarr_affinity_from_new_memory = MethodType(blended_cache, model)
    try:
        yield
    finally:
        model._get_image_feature = get_feature
        model.add_new_mask = add_mask
        model._cache_tarr_affinity_from_new_memory = cache_score
