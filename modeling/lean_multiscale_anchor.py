"""CMSA-inspired multiscale context for the released Lean TAMP anchor score.

This uses Lean's own max-cosine anchor scorer on pooled versions of its saved
memory feature map. It is not CMSA-Net's trained multi-stage CMA module.
"""
import hashlib
from types import MethodType

import torch
import torch.nn.functional as F


def multiscale_anchor_score(model, memory, anchor_tokens, local_score, strength):
    """Blend local and pooled feature-to-anchor scores at original token locations."""
    if not 0 <= strength <= 1:
        raise ValueError('strength must lie in [0, 1]')
    if memory.ndim != 4 or memory.shape[0] != 1:
        raise ValueError('Expected one dense spatial memory map')
    height, width = memory.shape[-2:]
    if height % 4 or width % 4:
        raise ValueError('Memory dimensions must be divisible by four')
    local = local_score.detach().float().reshape(-1)
    if local.numel() != height * width or not torch.isfinite(local).all():
        raise ValueError('Invalid local anchor score')
    if strength == 0:
        return local
    values = []
    for scale in (2, 4):
        pooled = F.avg_pool2d(memory.float(), kernel_size=scale, stride=scale)
        tokens = pooled.flatten(2).permute(2, 0, 1)
        score = model._compute_anchor_similarity_maxcos(tokens, anchor_tokens)
        if score is None or score.numel() != (height // scale) * (width // scale):
            raise ValueError('Pooled Lean anchor score has incorrect shape')
        score = score.reshape(1, 1, height // scale, width // scale).float()
        values.append(F.interpolate(score, size=(height, width), mode='bilinear', align_corners=False).reshape(-1))
    context = (values[0] + values[1]) * 0.5
    result = (1 - strength) * local.to(context.device) + strength * context
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite multi-scale anchor score')
    return result


def install_multiscale_anchor(predictor, strength, selection_events=None):
    """Wrap Lean's cache writer; leave its TARR affinity and model weights intact."""
    if not 0 <= strength <= 1:
        raise ValueError('strength must lie in [0, 1]')
    original = predictor._cache_tarr_affinity_from_new_memory
    events = []

    def cache(self, current_out, output_dict, frame_idx, obj_idx, is_init_cond_frame, run_mem_encoder, reverse):
        original(current_out, output_dict, frame_idx, obj_idx, is_init_cond_frame, run_mem_encoder, reverse)
        local = current_out.get('anchor_similarity_cache') if isinstance(current_out, dict) else None
        mgr = output_dict.get('anchor_mgr') if isinstance(output_dict, dict) else None
        memory = current_out.get('maskmem_features') if isinstance(current_out, dict) else None
        if strength == 0 or local is None:
            return
        if not isinstance(mgr, dict) or not torch.is_tensor(mgr.get('anchor_tokens')) or not torch.is_tensor(memory):
            raise RuntimeError('Lean anchor cache exists without its memory or anchors')
        with torch.no_grad():
            anchor = mgr['anchor_tokens']
            transformed = multiscale_anchor_score(self, memory.to(anchor.device), anchor, local.to(anchor.device), strength)
            current_out['anchor_similarity_cache'] = transformed.to(device=local.device, dtype=local.dtype)
            events.append({'frame_index': int(frame_idx), 'object_index': int(obj_idx),
                           'tokens': int(local.numel()),
                           'mean_abs_score_change': float((transformed - local.to(transformed.device)).abs().mean().item())})

    predictor._cache_tarr_affinity_from_new_memory = MethodType(cache, predictor)
    if selection_events is not None:
        original_dump = predictor._dump_retained_tokens_record

        def record_selection(self, **kwargs):
            original_dump(**kwargs)
            usage = kwargs['usage_scores'].reshape(-1)
            selected = kwargs['keep_idx'].detach().reshape(-1).long()
            main = kwargs['main_keep_idx'].detach().reshape(-1).long()
            count = int(usage.numel())
            expected = max(1, min(int(count * self.mem_keep_ratio), count))
            if main.numel() != expected or selected.numel() != expected or torch.unique(selected).numel() != expected:
                raise RuntimeError('Lean retained-token budget changed')
            if selected.numel() and (int(selected.min()) < 0 or int(selected.max()) >= count):
                raise RuntimeError('Lean selected a token outside its spatial memory segment')
            indices = torch.sort(selected).values.cpu().contiguous().numpy()
            source = kwargs['latest_prev'].get('frame_idx')
            selection_events.append({'frame_index': int(kwargs['frame_idx']),
                                     'source_frame_index': None if source is None else int(source),
                                     'candidates': count, 'kept': int(selected.numel()),
                                     'indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest()})

        predictor._dump_retained_tokens_record = MethodType(record_selection, predictor)
    return events
