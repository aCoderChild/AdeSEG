"""Distinguish exact tensor parity from unchanged binary mask decisions."""
import torch

from scripts.trace_key_adapter_parity import tensor_difference
from scripts.run_token_conditioned_key_adapter import _tree_digest


def test_small_logit_difference_does_not_imply_changed_binary_mask():
    left = torch.tensor([1.0, -1.0])
    right = torch.tensor([1.000001, -1.0])
    report = tensor_difference(left, right)
    assert not report['equal']
    assert report['changed_values'] == 1
    assert report['max_absolute_difference'] > 0
    assert report['positive_value_disagreements'] == 0


def test_signed_zero_is_reported_as_bitwise_difference():
    report = tensor_difference(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert report['equal']
    assert not report['bitwise_equal']


def test_equal_tensor_has_no_changes():
    value = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
    report = tensor_difference(value, value.clone())
    assert report['equal'] and report['bitwise_equal']
    assert report['changed_values'] == 0


def test_dtype_difference_is_not_silently_coerced():
    report = tensor_difference(torch.ones(2), torch.ones(2, dtype=torch.float64))
    assert not report['equal']
    assert report['left_dtype'] != report['right_dtype']


def test_native_position_encoding_list_is_compared_without_flattening():
    value = [torch.zeros(1, 64, 64, 64)]
    report = tensor_difference(value, [value[0].clone()])
    assert report['equal'] and report['bitwise_equal']
    changed = [value[0].clone()]
    changed[0][0, 0, 0, 0] = 1
    report = tensor_difference(value, changed)
    assert not report['equal']
    assert report['children'][0]['changed_values'] == 1


def test_position_encoding_container_type_is_not_ignored():
    report = tensor_difference([torch.zeros(1)], (torch.zeros(1),))
    assert report['structure_mismatch']
    assert not report['equal']


def test_scalar_optimizer_step_is_hashable_and_value_sensitive():
    state = {'state': {0: {'step': torch.tensor(17.0), 'exp_avg': torch.ones(2)}}}
    copied = {'state': {0: {'step': torch.tensor(17.0), 'exp_avg': torch.ones(2)}}}
    assert _tree_digest(state) == _tree_digest(copied)
    copied['state'][0]['step'].add_(1)
    assert _tree_digest(state) != _tree_digest(copied)


def test_scalar_digest_retains_shape_dtype_and_signed_zero():
    scalar = torch.tensor(0.0)
    assert _tree_digest(scalar) != _tree_digest(scalar.reshape(1))
    assert _tree_digest(scalar) != _tree_digest(scalar.double())
    assert _tree_digest(scalar) != _tree_digest(torch.tensor(-0.0))
    report = tensor_difference(scalar, torch.tensor(-0.0))
    assert report['equal'] and not report['bitwise_equal']
