"""Source-compatible scoped integration of early-FPN memory enrichment.

Only native memory tensors are enriched. Native decoder logits, object pointers,
position encodings, TAMP indices, TCIM and TARR remain owned by Lean-SAM2.
No patch is installed unless the caller explicitly enters this context.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
from types import MethodType

import torch
import torch.nn.functional as F

from modeling.lean_memory_feature_fusion import EarlyFeatureMemoryFusion


@contextmanager
def route_memory_feature_fusion(predictor, fusion: EarlyFeatureMemoryFusion, *,
                                identity_control: bool = False,
                                include_conditioning: bool = False,
                                telemetry: list | None = None,
                                input_mode: str = 'full'):
    """Temporarily enrich native _encode_new_memory output during track_step.

    Enrichment is applied to non-conditioning frames by default. Direct
    _run_memory_encoder calls outside track_step retain native behavior,
    including post-click corrections. This is an explicit one-shot causal
    ablation interface; simultaneous interaction/routing hooks are rejected.
    """
    if not isinstance(fusion, EarlyFeatureMemoryFusion):
        raise TypeError('Expected the source-bound memory fusion adapter')
    if (type(identity_control) is not bool or type(include_conditioning) is not bool
            or (telemetry is not None and not isinstance(telemetry, list))):
        raise ValueError('Invalid memory fusion routing options')
    if input_mode not in fusion.INPUT_MODES:
        raise ValueError('Invalid memory fusion input mode')
    if predictor.training or fusion.training:
        raise ValueError('Memory fusion inference requires both modules in eval mode')
    if identity_control:
        if fusion._trained_checkpoint_loaded or not fusion.is_identity_initialized():
            raise ValueError('Identity-control routing requires an untrained zero residual')
    elif not fusion._trained_checkpoint_loaded:
        raise ValueError('Learned memory fusion requires a validated trained checkpoint')

    # Do not compose with logit-routing wrappers or other hooks at this stage:
    # both paths alter what is written to future memory and need an explicit
    # factorial experiment before concurrent installation can be approved.
    forbidden = ('track_step', '_encode_new_memory', '_run_single_frame_inference',
                 '_track_step', '_encode_memory_in_output')
    if any(name in vars(predictor) for name in forbidden):
        raise RuntimeError('Conflicting predictor hooks are already installed')
    track = getattr(predictor, 'track_step', None)
    encode = getattr(predictor, '_encode_new_memory', None)
    if not callable(track) or not callable(encode):
        raise TypeError('Predictor lacks the pinned Lean-SAM2 memory API')
    track_sig = inspect.signature(track)
    encode_sig = inspect.signature(encode)
    required = ('frame_idx', 'is_init_cond_frame')
    encoder_args = ('current_vision_feats', 'feat_sizes', 'pred_masks_high_res',
                    'object_score_logits')
    if (any(x not in track_sig.parameters for x in required)
            or any(x not in encode_sig.parameters for x in encoder_args)):
        raise RuntimeError('Pinned Lean-SAM2 method signatures have changed')

    context = {'active': False, 'conditioning': False, 'frame': None,
               'anchor_mgr': None}

    def wrapped_track(self, *args, **kwargs):
        if context['active']:
            raise RuntimeError('Nested or concurrent memory fusion tracking is unsupported')
        bound = track_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        conditional = bound.arguments['is_init_cond_frame']
        if type(conditional) is not bool:
            raise ValueError('is_init_cond_frame must be explicitly boolean')
        output_dict = bound.arguments.get('output_dict')
        anchor_mgr = output_dict.get('anchor_mgr') if isinstance(output_dict, dict) else None
        context.update(active=True, conditioning=conditional,
                       frame=bound.arguments['frame_idx'], anchor_mgr=anchor_mgr)
        try:
            return track(*args, **kwargs)
        finally:
            context.update(active=False, conditioning=False, frame=None,
                           anchor_mgr=None)

    def wrapped_encode(self, *args, **kwargs):
        # Always let the pretrained memory encoder build its actual
        # mask-conditioned representation before applying any learnable delta.
        native, native_position = encode(*args, **kwargs)
        if not context['active'] or (context['conditioning'] and not include_conditioning):
            return native, native_position
        bound = encode_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        feats, sizes = a['current_vision_feats'], a['feat_sizes']
        if not isinstance(feats, (tuple, list)) or not isinstance(sizes, (tuple, list)):
            raise ValueError('Missing native Lean FPN levels')
        n = len(fusion.config.early_channels)
        if len(feats) != n + 1 or len(sizes) != len(feats):
            raise ValueError('Configured early level count disagrees with native FPN')
        if not isinstance(native, torch.Tensor) or native.ndim != 4:
            raise ValueError('Native memory encoder did not return spatial memory')
        batch = native.shape[0]
        early = []
        for i in range(n):
            fmap, hw = feats[i], sizes[i]
            if (not isinstance(fmap, torch.Tensor) or fmap.ndim != 3
                    or fmap.shape[1] != batch or len(hw) != 2
                    or fmap.shape[0] != int(hw[0]) * int(hw[1])):
                raise ValueError('Native FPN token-to-spatial mapping mismatch')
            early.append(fmap.permute(1, 2, 0).reshape(batch, fmap.shape[2], *hw))
        enriched = fusion(early, native, a['pred_masks_high_res'],
                          a['object_score_logits'], input_mode=input_mode)
        if enriched.shape != native.shape or enriched.dtype != native.dtype:
            raise RuntimeError('Memory fusion changed native token geometry')
        if telemetry is not None:
            # Lean-SAM2 stores spatial memory as BF16 in the video predictor.
            # Pre-conversion deltas can round to zero; measure both explicitly.
            with torch.no_grad():
                native_stored = native.detach().to(torch.bfloat16)
                enriched_stored = enriched.detach().to(torch.bfloat16)
                changed = enriched_stored != native_stored
                stored_delta = enriched_stored.float() - native_stored.float()
                changed_spatial = torch.any(changed, dim=1)
                before_norm = native.detach().float().square().mean().sqrt()
                # Compare memory-space target similarity using the same fixed
                # conditioning-frame anchors. This is NOT a TAMP selection
                # measurement: attention-usage scores arrive on a later read.
                anchor_stats = {
                    'anchor_reference_available': False,
                    'anchor_similarity_native_mean': None,
                    'anchor_similarity_enriched_mean': None,
                    'anchor_similarity_mean_abs_shift': None,
                }
                mgr = context['anchor_mgr']
                anchors = mgr.get('anchor_tokens') if isinstance(mgr, dict) else None
                if (isinstance(anchors, torch.Tensor) and anchors.ndim == 2
                        and anchors.shape[0] > 0 and native.shape[0] == 1
                        and anchors.shape[1] == native.shape[1]):
                    refs = F.normalize(anchors.detach().float().to(native.device),
                                       dim=1, eps=1e-6)
                    x0 = F.normalize(native_stored.flatten(2)[0].float().T,
                                     dim=1, eps=1e-6)
                    x1 = F.normalize(enriched_stored.flatten(2)[0].float().T,
                                     dim=1, eps=1e-6)
                    sim_native = (x0 @ refs.T).amax(dim=1)
                    sim_enriched = (x1 @ refs.T).amax(dim=1)
                    anchor_stats = {
                        'anchor_reference_available': True,
                        'anchor_similarity_native_mean': float(sim_native.mean().item()),
                        'anchor_similarity_enriched_mean': float(sim_enriched.mean().item()),
                        'anchor_similarity_mean_abs_shift': float(
                            (sim_enriched - sim_native).abs().mean().item()),
                    }
                telemetry.append({
                    'frame_idx': int(context['frame']),
                    'conditioning': bool(context['conditioning']),
                    'fusion_input_mode': input_mode,
                    **anchor_stats,
                    'native_token_count': int(native.shape[-2] * native.shape[-1]),
                    'enriched_token_count': int(enriched.shape[-2] * enriched.shape[-1]),
                    'changed_elements': int(torch.count_nonzero(enriched != native).item()),
                    'changed_elements_after_bf16': int(changed.sum().item()),
                    'changed_spatial_positions_after_bf16': int(changed_spatial.sum().item()),
                    'fraction_spatial_positions_changed_after_bf16': float(
                        changed_spatial.float().mean().item()
                    ),
                    'mean_abs_delta_after_bf16': float(stored_delta.abs().mean().item()),
                    'relative_l2_delta_after_bf16': float(
                        stored_delta.square().mean().sqrt().div(before_norm.clamp_min(1e-6)).item()
                    ),
                    'native_memory_rms': float(before_norm.item()),
                    'early_fpn_rms': [
                        float(e.detach().float().square().mean().sqrt().item()) for e in early
                    ],
                })
        return enriched, native_position

    # The predictor owns its methods again immediately after leaving the scope.
    try:
        predictor.track_step = MethodType(wrapped_track, predictor)
        predictor._encode_new_memory = MethodType(wrapped_encode, predictor)
        yield
    finally:
        altered = (
            getattr(getattr(predictor, 'track_step', None), '__func__', None) is not wrapped_track
            or getattr(getattr(predictor, '_encode_new_memory', None), '__func__', None)
            is not wrapped_encode
        )
        if getattr(getattr(predictor, 'track_step', None), '__func__', None) is wrapped_track:
            delattr(predictor, 'track_step')
        if getattr(getattr(predictor, '_encode_new_memory', None), '__func__', None) is wrapped_encode:
            delattr(predictor, '_encode_new_memory')
        if altered:
            raise RuntimeError('Predictor methods were unexpectedly replaced during fusion')


__all__ = ['route_memory_feature_fusion']
