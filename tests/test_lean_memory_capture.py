import pytest
import torch
from modeling.lean_memory_capture import clone_data, zero_memory_segment

def test_zero_spatial_content_preserves_pointer_and_original():
    original = torch.arange(4103 * 2).reshape(4103, 1, 2).float()
    before = original.clone()
    cond = zero_memory_segment(original, 3, 'conditioning')
    other = zero_memory_segment(original, 3, 'other_spatial')
    assert torch.equal(original, before)
    assert torch.count_nonzero(cond[:4096]) == 0
    assert torch.equal(cond[4096:], original[4096:])
    assert torch.equal(other[:4096], original[:4096])
    assert torch.count_nonzero(other[4096:-3]) == 0
    assert torch.equal(other[-3:], original[-3:])
    assert torch.equal(zero_memory_segment(original, 3, 'noop'), original)

def test_clone_is_independent_and_invalid_inputs_fail():
    x = torch.tensor([1.])
    copy = clone_data({'x': [x, None]})
    x.add_(1)
    assert copy['x'][0].item() == 1
    with pytest.raises(ValueError):
        zero_memory_segment(torch.zeros(4095, 1, 2), 0, 'conditioning')
    with pytest.raises(ValueError):
        zero_memory_segment(torch.zeros(4096, 1, 2), 0, 'unknown')
