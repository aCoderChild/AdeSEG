"""Read-only value interventions using native SAM2 and the existing MAVOS hook.

GT teachers are explicit TRAIN oracle inputs, not a deployable model. Native
keys/positions/pointers stay fixed; later-layer queries may change with values.
"""
from contextlib import ExitStack

import torch

from modeling.complementary_memory.integration import NativeComplementaryRead
from modeling.mavos_focal_value_modulation import MAVOSMemoryValueModulation
from modeling.memory_value_oracle import FixedSpatialValues
from scripts.run_token_conditioned_key_adapter import _tree_digest

MEMORY_FIELDS = ('maskmem_pos_enc', 'pred_masks', 'obj_ptr', 'object_score_logits', 'maskmem_features')


def spatial_values(bank, spans, teachers=None, arm='identity_values'):
    if arm not in ('identity_values', 'mask_only', 'mask_and_presence'):
        raise ValueError('Unknown fixed value intervention')
    blocks = []
    cursor = 0
    for span in spans:
        if span['start'] != cursor:
            raise RuntimeError('Memory spans must be contiguous in native order')
        frame = span['frame']
        conditioned = frame in bank['cond_frame_outputs']
        stored = bank['cond_frame_outputs'][frame] if conditioned else bank['non_cond_frame_outputs'][frame]
        native = stored['maskmem_features']
        value = native if conditioned or arm == 'identity_values' else teachers[frame][arm]
        if value.shape != native.shape or value.dtype != native.dtype or value.device != native.device:
            raise RuntimeError('Teacher differs from source value shape/dtype/device')
        block = value.flatten(2).permute(2, 0, 1)
        if block.shape[0] != span['stop'] - span['start']:
            raise RuntimeError('Teacher token count differs from native span')
        blocks.append(block)
        cursor = span['stop']
    if not blocks:
        raise RuntimeError('No native spatial frame selected')
    return torch.cat(blocks)


def query(model, features, bank, frame, num_frames, expected=None, replacement=None):
    """Replay one query without adding its result to the supplied bank."""
    if (expected is None) != (replacement is None):
        raise ValueError('Both expected and replacement values are required')
    before = _tree_digest(bank)
    keys, handles = [], []
    try:
        for index, layer in enumerate(model.memory_attention.layers):
            def observe(module, args, index=index):
                if len(args) != 1:
                    raise RuntimeError('Unexpected native key projection arguments')
                keys.append({'layer': index, 'key_input_sha256': _tree_digest(args[0])})
            handles.append(layer.cross_attn_image.k_proj.register_forward_pre_hook(observe))
        with ExitStack() as stack, torch.inference_mode():
            observer = stack.enter_context(NativeComplementaryRead(model, None, trace_only=True))
            value_hook = None
            if expected is not None:
                adapters = [FixedSpatialValues(expected, replacement) for _ in range(4)]
                value_hook = stack.enter_context(MAVOSMemoryValueModulation(
                    model, adapters, record_value_receipts=True, record_modulation_stats=True))
            output = model.track_step(frame_idx=frame, is_init_cond_frame=False,
                current_vision_feats=features['vision'], current_vision_pos_embeds=features['positions'],
                feat_sizes=features['sizes'], point_inputs=None, mask_inputs=None,
                output_dict=bank, num_frames=num_frames, track_in_reverse=False, run_mem_encoder=True)
            stored = {key: output[key] for key in MEMORY_FIELDS}
            stored['maskmem_features'] = stored['maskmem_features'].to(torch.bfloat16)
            result = {'output': stored, 'output_sha256': _tree_digest(stored),
                'keys': keys, 'calls': list(observer.calls),
                'values': list(value_hook.value_calls) if value_hook else [],
                'value_changes': list(value_hook.modulation_stats) if value_hook else []}
    finally:
        for handle in handles:
            handle.remove()
    if _tree_digest(bank) != before:
        raise RuntimeError('Read-only oracle query mutated its native bank')
    if [k['layer'] for k in keys] != [0, 1, 2, 3]:
        raise RuntimeError('Native key projection coverage changed')
    if len(result['calls']) != 1 or result['calls'][0]['layers'] != [0, 1, 2, 3]:
        raise RuntimeError('Incomplete four-layer native read')
    if expected is not None:
        if [v['layer'] for v in result['values']] != [0, 1, 2, 3]:
            raise RuntimeError('Incomplete value intervention')
        if any(not v['pointer_tail_bit_identical'] for v in result['values']):
            raise RuntimeError('Pointer values changed')
    return result
