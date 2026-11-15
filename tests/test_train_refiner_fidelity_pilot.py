import pytest
import torch

from scripts.train_refiner_fidelity_pilot import WEIGHT, fidelity_loss
from scripts.train_multilevel_refiner_seq4_empty_v2 import loss_fn


def test_fidelity_is_zero_for_unchanged_prediction():
    original = torch.tensor([[[[-2.0, 0.0], [1.0, 3.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    torch.testing.assert_close(fidelity_loss(original, target, original), loss_fn(original, target))


def test_fidelity_uses_original_as_detached_reference():
    original = torch.zeros((1, 1, 2, 2), requires_grad=True)
    prediction = torch.full((1, 1, 2, 2), 2.0, requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = fidelity_loss(prediction, target, original)
    plain = loss_fn(prediction, target)
    assert loss > plain and WEIGHT == 0.1
    loss.backward()
    assert prediction.grad is not None and original.grad is None


def test_fidelity_rejects_misaligned_inputs():
    prediction = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError, match='geometry'):
        fidelity_loss(prediction, prediction, torch.zeros(1, 1, 1, 1))
