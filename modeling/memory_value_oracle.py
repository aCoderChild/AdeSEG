"""Fixed, loss-free value replacement for an explicitly labelled oracle diagnostic.

Reuses MAVOSMemoryValueModulation's existing native v_proj integration. This
contains no learned parameters and is not a deployable segmentation method.
"""
import torch
from torch import nn


class FixedSpatialValues(nn.Module):
    def __init__(self, expected, replacement, grid_shape=(64, 64)):
        super().__init__()
        if expected.ndim != 3 or expected.shape != replacement.shape or expected.shape[1] != 1:
            raise ValueError('Expected aligned single-object [spatial_tokens,1,channels] values')
        if expected.shape[0] == 0 or expected.shape[0] % (grid_shape[0] * grid_shape[1]):
            raise ValueError('Expected complete spatial frame blocks')
        if expected.dtype != replacement.dtype or expected.device != replacement.device:
            raise ValueError('Expected reference and replacement in the same bank dtype/device')
        if not torch.isfinite(expected).all() or not torch.isfinite(replacement).all():
            raise ValueError('Oracle values must be finite')
        self.channels = expected.shape[-1]
        self.grid_shape = tuple(grid_shape)
        self.register_buffer('expected', expected.detach().clone())
        self.register_buffer('replacement', replacement.detach().clone())

    def forward(self, memory, pointer_tokens=0):
        if memory.ndim != 3 or not 0 <= pointer_tokens < memory.shape[0]:
            raise RuntimeError('Invalid sequence-first memory/pointer layout')
        spatial = memory.shape[0] - pointer_tokens
        expected = self.expected.to(dtype=memory.dtype)
        if memory[:spatial].shape != expected.shape or not torch.equal(memory[:spatial], expected):
            raise RuntimeError('Oracle reference does not align with actual native value content')
        replacement = self.replacement.to(dtype=memory.dtype)
        if torch.equal(replacement, expected):
            return memory  # Exact identity also preserves native storage/layout.
        result = memory.clone()
        result[:spatial] = replacement
        if not torch.equal(result[spatial:], memory[spatial:]):
            raise RuntimeError('Oracle modified object-pointer values')
        return result
