"""Exploratory factorial routing with observer-only logit and presence telemetry.

The predictor and Lean memory selection are unchanged. B changes only the
returned mask, C only the mask sent to the native memory encoder, and D both.
The initial conditioning output is protected. This adapter is intended for
controlled evaluation, not concurrent or interactive predictor sessions.
"""
from contextlib import contextmanager
import inspect
import hashlib
from types import MethodType

import torch
import torch.nn.functional as F

from modeling.contrast_refinement_routing import route_refined_logits
from modeling.multilevel_contrast_refiner import MultilevelContrastRefiner


@contextmanager
def route_boundary_refinement(model, refiner, arm, *, test_only=False, runtime_lease=None, telemetry=None):
    """Install the refiner for a single-object, one-shot propagation session.

    A is the native control. Synthetic stimuli require ``test_only=True``.
    A learned refiner requires a live, verifier-owned runtime lease. A lease
    cannot by itself establish independent test data or clinical efficacy.
    """
    if arm not in ('A', 'B', 'C', 'D'):
        raise ValueError('Invalid refinement arm')
    if type(test_only) is not bool:
        raise ValueError('test_only must be a boolean')
    if arm == 'A' and runtime_lease is not None:
        from modeling.lean_verified_runtime_lease import _RuntimeLease
        if test_only or not isinstance(refiner, MultilevelContrastRefiner) or not isinstance(runtime_lease, _RuntimeLease):
            raise RuntimeError('Baseline runtime provenance requires a verifier-owned runtime lease')
        runtime_lease.validate_live(model, refiner, tensors=True)
    if arm != 'A':
        if test_only:
            if isinstance(refiner, MultilevelContrastRefiner) or runtime_lease is not None:
                raise RuntimeError('Synthetic test mode cannot authorize learned-refiner runtime provenance')
        else:
            from modeling.lean_verified_runtime_lease import _RuntimeLease
            if not isinstance(refiner, MultilevelContrastRefiner) or not isinstance(runtime_lease, _RuntimeLease):
                raise RuntimeError('Learned-refiner runtime provenance requires a verifier-owned runtime lease')
            runtime_lease.validate_live(model, refiner, tensors=True)
    if model.training:
        raise ValueError('Predictor must be in evaluation mode')
    if getattr(model, 'fill_hole_area', None) != 0:
        raise ValueError('This routing requires native hole filling to be disabled')
    if arm != 'A' and (refiner is None or refiner.training or
                       not getattr(refiner, '_trained_checkpoint_loaded', False)):
        raise ValueError('An evaluated, provenance-checked refiner is required')

    names = ('_run_single_frame_inference', '_track_step', '_encode_memory_in_output')
    if any(name in vars(model) for name in names):
        raise RuntimeError('Cannot compose with existing instance-level method hooks')
    originals = {name: getattr(model, name) for name in names}
    sigs = {name: inspect.signature(method) for name, method in originals.items()}
    state = {'active': False, 'frame': None, 'computed': False, 'encoded': False,
             'original_low': None, 'original_high': None, 'output': None, 'memory': None}

    def track(self, *args, **kwargs):
        if not state['active'] or state['computed']:
            raise RuntimeError('Unexpected or repeated track step')
        bound = sigs['_track_step'].bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if a['frame_idx'] != state['frame'] or a['is_init_cond_frame'] != state['conditioning']:
            raise RuntimeError('Track step frame or conditioning mismatch')
        result = originals['_track_step'](*args, **kwargs)
        current, sam_outputs, early_features, deep = result
        low, high = sam_outputs[3], sam_outputs[4]
        state.update(computed=True, original_low=low, original_high=high)
        if state['conditioning'] or arm == 'A':
            state['output'], state['memory'] = low, high
            if telemetry is not None:
                telemetry.append(_stats(a['frame_idx'], bool(state['conditioning']), sam_outputs[6], low, low, low, high))
            return result
        if (a['point_inputs'] is not None or a['mask_inputs'] is not None or
                a['prev_sam_mask_logits'] is not None):
            raise ValueError('Interactive corrections are outside the one-shot protocol')
        if not isinstance(early_features, (tuple, list)) or not early_features:
            raise ValueError('Missing early SAM feature map')
        with torch.inference_mode():
            refined_low = refiner(early_features[0].clone(), deep.clone(), low.clone())
        out, selected_low = route_refined_logits(low, refined_low, arm, conditioning=False)
        # Native high-res mask must remain bit-identical for identity refinement.
        if selected_low is low or torch.equal(selected_low, low):
            selected_high = high
        else:
            selected_high = F.interpolate(selected_low, size=high.shape[-2:], mode='bilinear', align_corners=False)
            if selected_high.shape != high.shape or selected_high.dtype != high.dtype or not torch.isfinite(selected_high).all():
                raise ValueError('Refined memory logits do not match native geometry')
        state['output'], state['memory'] = out, selected_high
        if telemetry is not None:
            telemetry.append(_stats(a['frame_idx'], False, sam_outputs[6], low, refined_low, out, selected_high))
        return current, sam_outputs, early_features, deep

    def encode(self, *args, **kwargs):
        if not state['active'] or not state['computed'] or state['encoded']:
            raise RuntimeError('Unexpected or repeated memory encoding')
        bound = sigs['_encode_memory_in_output'].bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments['high_res_masks'] is not state['original_high']:
            raise RuntimeError('Memory mask is not the current decoder output')
        state['encoded'] = True
        bound.arguments['high_res_masks'] = state['memory']
        return originals['_encode_memory_in_output'](*bound.args, **bound.kwargs)

    def run(self, *args, **kwargs):
        if (getattr(getattr(self, '_track_step'), '__func__', None) is not track or
                getattr(getattr(self, '_encode_memory_in_output'), '__func__', None) is not encode):
            raise RuntimeError('Active refinement routing hook changed')
        if runtime_lease is not None and not test_only:
            runtime_lease.validate_live(self, refiner, tensors=False)
        if state['active']:
            raise RuntimeError('Concurrent or nested inference is unsupported')
        bound = sigs['_run_single_frame_inference'].bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if a['batch_size'] != 1 or a['obj_idx'] != 0:
            raise ValueError('Only single-object propagation is supported')
        ids = a['inference_state'].get('obj_ids')
        if ids is not None and len(ids) != 1:
            raise ValueError('Only single-object propagation is supported')
        conditioning = a['is_init_cond_frame']
        if type(conditioning) is not bool:
            raise ValueError('Conditioning must be explicit')
        if not conditioning and (a.get('store_as_cond', False) or a['point_inputs'] is not None or
                                 a['mask_inputs'] is not None or a['prev_sam_mask_logits'] is not None):
            raise ValueError('Interactive corrections are outside the one-shot protocol')
        state.update(active=True, frame=a['frame_idx'], conditioning=conditioning,
                     computed=False, encoded=False, original_low=None, original_high=None,
                     output=None, memory=None)
        try:
            compact, native_output = originals['_run_single_frame_inference'](*args, **kwargs)
            if not state['computed'] or not state['encoded']:
                raise RuntimeError('Predictor did not execute the expected routing hooks')
            if native_output is not state['original_low']:
                raise RuntimeError('Native output was unexpectedly transformed')
            return compact, state['output']
        finally:
            state.update(active=False, frame=None, original_low=None, original_high=None,
                         output=None, memory=None)

    try:
        model._run_single_frame_inference = MethodType(run, model)
        model._track_step = MethodType(track, model)
        model._encode_memory_in_output = MethodType(encode, model)
        yield
    finally:
        for name in names:
            if name in vars(model):
                delattr(model, name)


def _stats(frame, conditioning, object_score, native, refined, output, memory):
    """Record observables only. Zero here is SAM2's existing mask convention."""
    if object_score.numel() != 1 or not torch.isfinite(object_score).all():
        raise ValueError('Unexpected object-presence tensor')
    def summarize(value):
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite mask logits')
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return {'min': float(value.min().item()), 'max': float(value.max().item()),
                'positive_pixels': int((value > 0).sum().item()), 'numel': value.numel(),
                'sha256': hashlib.sha256(raw.numpy().tobytes()).hexdigest()}
    score = float(object_score.item())
    return {'frame': int(frame), 'conditioning': conditioning, 'object_score_logit': score,
            'native_object_present': score > 0,
            'native_exact_suppressed': bool(torch.all(native == -1024).item()),
            'native': summarize(native), 'refined': summarize(refined),
            'output': summarize(output), 'memory_high_res': summarize(memory)}
