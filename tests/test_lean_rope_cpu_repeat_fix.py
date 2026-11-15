from types import SimpleNamespace

import pytest
import torch

from modeling.lean_rope_cpu_repeat_fix import corrected_cpu_key_frequency_repeat


def buggy(freqs, repeats):
    if repeats == 1:
        return freqs
    return freqs.unsqueeze(1).expand(-1, repeats, -1).reshape(-1, freqs.shape[-1])


def test_cpu_repeats_entire_complex_grid_and_preserves_single_repeat():
    module = SimpleNamespace(_repeat_freqs_for_k=buggy)
    base = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j], [9+10j, 11+12j]])
    with corrected_cpu_key_frequency_repeat(module):
        assert module._repeat_freqs_for_k(base, 1) is base
        for repeats in (2, 3, 6):
            assert torch.equal(module._repeat_freqs_for_k(base, repeats), base.repeat(repeats, 1))
    assert module._repeat_freqs_for_k is buggy


def test_scoped_patch_is_restored_after_error():
    module = SimpleNamespace(_repeat_freqs_for_k=buggy)
    with pytest.raises(RuntimeError):
        with corrected_cpu_key_frequency_repeat(module):
            raise RuntimeError('fixture')
    assert module._repeat_freqs_for_k is buggy
