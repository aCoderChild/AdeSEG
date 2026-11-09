import pytest
import torch

from modeling.memory_value_oracle import FixedSpatialValues


def test_identity_returns_native_tensor_and_has_no_trainable_parameters():
    spatial = torch.randn(32, 1, 8)
    adapter = FixedSpatialValues(spatial, spatial.clone(), (4, 4))
    memory = torch.cat((spatial, torch.randn(2, 1, 8)))
    assert adapter(memory, 2) is memory
    assert not list(adapter.parameters())


def test_only_spatial_values_change_without_mutating_bank_or_pointers():
    spatial = torch.randn(32, 1, 8).bfloat16()
    replacement = spatial + 1
    adapter = FixedSpatialValues(spatial, replacement, (4, 4))
    memory = torch.cat((spatial.float(), torch.randn(2, 1, 8)))
    saved = memory.clone()
    output = adapter(memory, 2)
    assert torch.equal(output[:32], replacement.float())
    assert torch.equal(output[32:], saved[32:])
    assert torch.equal(memory, saved)


def test_wrong_source_values_fail_instead_of_silently_replacing():
    spatial = torch.randn(16, 1, 8)
    adapter = FixedSpatialValues(spatial, spatial + 1, (4, 4))
    with pytest.raises(RuntimeError):
        adapter(spatial + .1)


def test_wrong_frame_or_pointer_span_fails():
    spatial = torch.randn(16, 1, 8)
    adapter = FixedSpatialValues(spatial, spatial + 1, (4, 4))
    with pytest.raises(RuntimeError):
        adapter(spatial, 1)


def test_partial_grid_and_nonfinite_teacher_fail():
    with pytest.raises(ValueError):
        FixedSpatialValues(torch.zeros(15, 1, 8), torch.zeros(15, 1, 8), (4, 4))
    bad = torch.full((16, 1, 8), float('nan'))
    with pytest.raises(ValueError):
        FixedSpatialValues(torch.zeros_like(bad), bad, (4, 4))
