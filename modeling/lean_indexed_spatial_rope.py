"""Use retained grid locations for Lean's existing rotary cross-attention.

Scoped to the audited CPU/FP32, single-object model with no attention KV
merging. It leaves query/key projections, attention, memory selection, pointer
tail, model weights, and author source untouched. This is experimental.
"""
from contextlib import contextmanager
from types import MethodType
import hashlib

import torch

from modeling.lean_memory_position_lookup import MemoryPositionLookup


@contextmanager
def indexed_spatial_rope(model, observations):
    from sam2.modeling import memory_attention as module

    if model.device.type != 'cpu' or any(layer.toma_drop_ratio_ca_kv != 0 for layer in model.memory_attention.layers):
        raise ValueError('Indexed RoPE supports only CPU and unmerged attention keys')
    with torch.no_grad():
        grid = model.memory_encoder.position_encoding(torch.zeros(1, model.mem_dim, 64, 64)).flatten(2).permute(2, 0, 1)[:, 0, :].contiguous()
        offsets = model.maskmem_tpos_enc.detach().reshape(-1, model.mem_dim).cpu()
        lookup = MemoryPositionLookup(grid, offsets)
        freq = [layer.cross_attn_image.compute_cis(end_x=64, end_y=64).cpu() for layer in model.memory_attention.layers]
        if any(not torch.equal(x, freq[0]) for x in freq[1:]):
            raise ValueError('Attention layers use different rotary grids')
        full_freq = freq[0]

    attention = model.memory_attention
    original_forward = attention.forward

    def forward(self, *args, **kwargs):
        curr = kwargs['curr']
        curr = curr[0] if isinstance(curr, list) else curr
        ptr = int(kwargs['num_obj_ptr_tokens'])
        if curr.shape[0] != full_freq.shape[0] or curr.shape[1] != 1:
            raise ValueError('Unexpected query grid or batch size')
        indices = lookup.decode(kwargs['memory_pos'], ptr)
        if not len(indices):
            raise ValueError('No spatial memory positions')
        old_query, old_key = module._apply_rotary_repeat_2d, module._repeat_freqs_for_k
        calls = {'query': 0, 'key': 0}

        def rotate_query(x, _base, _length):
            if x.shape[-2] != full_freq.shape[0]:
                raise ValueError('Query token count changed')
            calls['query'] += 1
            return old_query(x, full_freq, full_freq.shape[0])

        def key_frequencies(base, repeats):
            if base.shape[0] * repeats != len(indices) or base.shape[1] != full_freq.shape[1]:
                raise ValueError('Spatial key count changed')
            calls['key'] += 1
            return full_freq[indices]

        module._apply_rotary_repeat_2d = rotate_query
        module._repeat_freqs_for_k = key_frequencies
        try:
            output = original_forward(*args, **kwargs)
        finally:
            module._apply_rotary_repeat_2d = old_query
            module._repeat_freqs_for_k = old_key
        expected = len(self.layers)
        if calls != {'query': expected, 'key': expected}:
            raise ValueError(f'Unexpected rotary calls: {calls}')
        observations.append({'spatial_keys': len(indices), 'pointer_keys': ptr,
                             'indices_sha256': hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                             'query_rotations': calls['query'], 'key_rotations': calls['key']})
        return output

    attention.forward = MethodType(forward, attention)
    try:
        yield
    finally:
        attention.forward = original_forward
