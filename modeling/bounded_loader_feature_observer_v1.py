"""Read-only minimal decoder-input observer for the bounded-loader gate.

Adapted from `lean_mps_decoder_observer.py`: save only early and deep decoder
inputs for selected frames, not the much larger complete snapshots.
"""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch

from scripts.memory_token_io import digest


@contextmanager
def observe_decoder_features(model, targets, output, records):
    targets = set(targets); state = {'frame': None, 'captured': set()}
    if any(name in vars(model) for name in ('_get_image_feature', '_forward_sam_heads')):
        raise RuntimeError('Decoder observer cannot replace existing instance hooks')
    original_feature = model._get_image_feature; original_head = model._forward_sam_heads
    def feature(self, inference_state, frame_idx, batch_size):
        result = original_feature(inference_state, frame_idx, batch_size)
        state['frame'] = int(frame_idx)
        return result
    def head(self, *args, **kwargs):
        if args: raise ValueError('Unexpected positional decoder arguments')
        frame = state['frame']
        if frame in targets and frame not in state['captured']:
            early = kwargs.get('high_res_features'); deep = kwargs.get('backbone_features')
            if not early or deep is None or early[0].shape != (1, 32, 256, 256) or deep.shape != (1, 256, 64, 64) or \
                    not torch.isfinite(early[0]).all() or not torch.isfinite(deep).all():
                raise ValueError('Missing or nonfinite decoder features')
            data = {'early': early[0].detach().cpu().clone(), 'deep': deep.detach().cpu().clone()}
            path = Path(output)/f'features_{frame:06d}.pt'; torch.save(data, path)
            records.append({'frame': frame, 'path': str(path), 'sha256': digest(path)})
            state['captured'].add(frame)
        return original_head(*args, **kwargs)
    try:
        model._get_image_feature = MethodType(feature, model); model._forward_sam_heads = MethodType(head, model)
        yield
        if state['captured'] != targets: raise ValueError('Decoder-feature coverage differs')
    finally:
        for name in ('_get_image_feature', '_forward_sam_heads'):
            if name in vars(model): delattr(model, name)
