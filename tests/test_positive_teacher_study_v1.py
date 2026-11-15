import pytest
import torch
from scripts.train_positive_teacher_v1 import ARMS, LOSSES, TEACHER_WEIGHT, objective, validate_parity_checks
from scripts.score_positive_teacher_v1 import validate_training_order


def test_matched_arms_and_objective():
    assert ARMS == ('region', 'global', 'mask_only')
    assert LOSSES == ('robust_bce', 'robust_teacher')
    pred = torch.tensor([[[[-1., 2.]]]], requires_grad=True)
    original = torch.tensor([[[[3., 3.]]]])
    target = torch.tensor([[[[1., 0.]]]])
    base, core, preservation = objective(pred, original, target, 'robust_bce')
    plus, other_core, other_preservation = objective(pred, original, target, 'robust_teacher')
    torch.testing.assert_close(base, core)
    torch.testing.assert_close(core, other_core)
    torch.testing.assert_close(plus-base, TEACHER_WEIGHT*preservation)
    torch.testing.assert_close(preservation, other_preservation)
    with pytest.raises(ValueError): objective(pred, original, target, 'bad')


def test_parity_coverage_and_training_order_reject_incomplete_data():
    with pytest.raises(ValueError): validate_parity_checks([])
    with pytest.raises(ValueError): validate_training_order([], {'samples': []})
