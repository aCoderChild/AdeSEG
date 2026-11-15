"""Capture native SAM2 reads for supervised, frozen-backbone gate training."""
import ast
from functools import lru_cache

import torch
import torch.nn.functional as F

from modeling.cfbi_region_gate import NativeCFBIGate
from modeling.region_tokens import SAM2_REVISION, _verified_source


@lru_cache(maxsize=1)
def _author_focal_loss():
    path, source = _verified_source('sam2', SAM2_REVISION, 'training/loss_fns.py')
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'sigmoid_focal_loss')
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['sigmoid_focal_loss']


def object_presence_loss(object_score, target):
    presence = target.flatten(1).gt(0).any(dim=1).to(object_score.dtype).unsqueeze(1)
    if object_score.shape != presence.shape:
        raise ValueError('Expected one object score per target mask')
    return _author_focal_loss()(object_score, presence, len(presence), alpha=-1, gamma=0)


def clone_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tensors(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_tensors(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_tensors(v) for v in value)
    return value


def capture_training_read(model, state, frame, *, object_id=None):
    """Copy the image features and earlier native memory used for this frame."""
    if frame not in state['cached_features']:
        raise ValueError('Capture requires the current frame already in the native image cache')
    _, _, vision, positions, sizes = model._get_image_feature(state, frame, 1)
    obj_idx = 0 if object_id is None else state['obj_id_to_idx'][object_id]
    source = state['output_dict_per_obj'][obj_idx]
    prior = {bucket: {i: value for i, value in source[bucket].items() if i < frame}
             for bucket in ('cond_frame_outputs', 'non_cond_frame_outputs')}
    if not prior['cond_frame_outputs']:
        raise ValueError('Missing earlier prompt memory')
    with torch.inference_mode(False):
        return clone_tensors({'frame': frame, 'num_frames': state['num_frames'],
                              'vision': vision, 'positions': positions, 'sizes': sizes,
                              'output_dict': prior})


def replay_training_read(model, gate, sample, mode, sequence, *, return_object_score=False):
    """Run author track_step without encoding or writing a new memory."""
    with NativeCFBIGate(model, gate, mode, sequence) as observer:
        output = model.track_step(
            frame_idx=sample['frame'], is_init_cond_frame=False,
            current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
            feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
            output_dict=sample['output_dict'], num_frames=sample['num_frames'],
            track_in_reverse=False, run_mem_encoder=False)
    if len(observer.trace) != 1 or observer.trace[0]['layers'] != [0, 1, 2, 3]:
        raise RuntimeError('Expected one read across four attention layers')
    if return_object_score:
        return output['pred_masks'], output['object_score_logits'], observer.trace[0]
    return output['pred_masks'], observer.trace[0]
