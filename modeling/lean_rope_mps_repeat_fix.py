"""Match Lean's released CUDA whole-grid rotary-frequency repeat on MPS.

This targeted intervention does not fix compressed-key coordinates or the
smaller repeated query grid used by the author code after compression.
"""
from contextlib import contextmanager
import torch


@contextmanager
def corrected_mps_key_frequency_repeat(module):
    original = module._repeat_freqs_for_k

    def corrected(freqs, repeats):
        if freqs.device.type != 'mps' or repeats <= 1:
            return original(freqs, repeats)
        return torch.cat([freqs] * repeats, dim=0)

    module._repeat_freqs_for_k = corrected
    try:
        yield
    finally:
        module._repeat_freqs_for_k = original
