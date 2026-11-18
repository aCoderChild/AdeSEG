"""Scoped correction for Lean-SAM2's CPU key-frequency repeat ordering.

The released CUDA branch repeats the complete frequency grid. Its CPU branch
interleaves individual rows instead. This adapter preserves the released
implementation for repeat=1 and non-CPU tensors, and uses ``torch.cat`` to
repeat the complete complex grid on CPU. It does not correct the separate
question of rotary coordinates for spatially pruned tokens.

The author module is patched process-wide only for the duration of the context;
use separate processes for concurrent baseline and corrected inference.
"""
from contextlib import contextmanager

import torch


@contextmanager
def corrected_cpu_key_frequency_repeat(memory_attention_module):
    """Temporarily match the released CUDA repeat order on CPU."""
    original = memory_attention_module._repeat_freqs_for_k

    def corrected(freqs_cis_base, k_repeat):
        if freqs_cis_base.device.type != 'cpu' or k_repeat <= 1:
            return original(freqs_cis_base, k_repeat)
        # Concatenate complete grids; unlike interleaving, this matches the
        # author's CUDA ``freqs_cis_base.repeat(k_repeat, 1)`` expression.
        return torch.cat([freqs_cis_base] * k_repeat, dim=0)

    memory_attention_module._repeat_freqs_for_k = corrected
    try:
        yield
    finally:
        memory_attention_module._repeat_freqs_for_k = original
