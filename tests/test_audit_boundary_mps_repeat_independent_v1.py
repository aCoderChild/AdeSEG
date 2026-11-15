import pytest
import torch

from scripts.audit_boundary_mps_repeat_independent_v1 import difference


def test_independent_nested_optimizer_difference():
    a = {'state': {0: {'step': torch.tensor(1.), 'moment': torch.tensor([1., 2.])}}, 'groups': [1, 2]}
    b = {'state': {0: {'step': torch.tensor(1.), 'moment': torch.tensor([1., 2.0001])}}, 'groups': [1, 2]}
    assert difference(a, a) == (True, 0.0)
    exact, delta = difference(a, b)
    assert not exact and delta == pytest.approx(0.0001, abs=1e-6)
    with pytest.raises(ValueError, match='keys'):
        difference(a, {'state': {}, 'groups': [1, 2]})


def test_nonfinite_and_tensor_metadata_fail_closed():
    with pytest.raises(ValueError, match='finiteness'):
        difference(torch.tensor([float('nan')]), torch.tensor([0.]))
    with pytest.raises(ValueError, match='metadata'):
        difference(torch.ones(2), torch.ones(3))
