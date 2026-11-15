import pytest
import torch

from modeling.lean_live_state_audit import assert_module_tensor_fingerprints, module_tensor_fingerprints


def test_parameter_and_nonpersistent_buffer_mutations_are_detected():
    model = torch.nn.Linear(2, 3)
    model.register_buffer('temporary_cache', torch.arange(3), persistent=False)
    expected = module_tensor_fingerprints(model)
    assert 'parameter:weight' in expected and 'buffer:temporary_cache' in expected
    assert_module_tensor_fingerprints(model, expected)
    model.temporary_cache[0] = 7
    with pytest.raises(ValueError, match='buffer:temporary_cache'):
        assert_module_tensor_fingerprints(model, expected)
    model.temporary_cache[0] = 0
    with torch.no_grad():
        model.weight[0, 0] += 1
    with pytest.raises(ValueError, match='parameter:weight'):
        assert_module_tensor_fingerprints(model, expected)


def test_byte_hash_preserves_bfloat_and_complex_dtype():
    model = torch.nn.Module()
    model.register_buffer('bfloat', torch.ones(2, dtype=torch.bfloat16))
    model.register_buffer('complex', torch.tensor([1+2j]))
    result = module_tensor_fingerprints(model)
    assert result['buffer:bfloat']['dtype'] == 'torch.bfloat16'
    assert result['buffer:complex']['dtype'] == 'torch.complex64'
    assert len(result['buffer:complex']['sha256']) == 64
