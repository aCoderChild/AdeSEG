import pytest
import torch
import torch.nn.functional as F

from modeling.training_foreground_preservation_v1 import expand_logits, positive_preservation_penalty


def test_expansion_matches_dense_max_and_preserves_suppression():
    generator = torch.Generator().manual_seed(91)
    logits = torch.randn((2, 1, 19, 23), generator=generator)
    for width in (1, 3, 9, 17):
        actual = expand_logits(logits, width)
        expected = F.max_pool2d(logits, width, stride=1, padding=width//2)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.all(actual >= logits)
    assert expand_logits(logits, 1) is logits
    assert torch.equal(expand_logits(torch.full((1, 1, 19, 23), -1024.), 17),
                       torch.full((1, 1, 19, 23), -1024.))
    for width in (0, 2, 25, True):
        with pytest.raises(ValueError):
            expand_logits(logits, width)


def test_only_true_foreground_probability_reduction_is_penalized():
    target = torch.tensor([[[[1., 1., 0., 0.]]]])
    teacher = torch.tensor([[[[2., 0., 4., 4.]]]], requires_grad=True)
    prediction = torch.tensor([[[[-2., 2., -4., 4.]]]], requires_grad=True)
    loss = positive_preservation_penalty(prediction, teacher, target)
    assert loss > 0
    loss.backward()
    assert prediction.grad[0, 0, 0, 0] < 0
    assert torch.count_nonzero(prediction.grad[0, 0, 0, 1:]) == 0
    assert teacher.grad is None


def test_empty_or_suppressed_teacher_and_increases_have_no_penalty():
    prediction = torch.zeros((1, 1, 3, 3), requires_grad=True)
    assert positive_preservation_penalty(prediction, torch.ones_like(prediction), torch.zeros_like(prediction)) == 0
    assert positive_preservation_penalty(prediction, torch.full_like(prediction, -1024), torch.ones_like(prediction)) == 0
    assert positive_preservation_penalty(prediction, torch.full_like(prediction, -2), torch.ones_like(prediction)) == 0
    assert positive_preservation_penalty(prediction, prediction.detach(), torch.ones_like(prediction)) == 0


def test_teacher_confidence_does_not_cancel_normalization():
    pred = torch.full((1, 1, 2, 2), -3.)
    target = torch.ones_like(pred)
    strong = positive_preservation_penalty(pred, torch.full_like(pred, 3.), target)
    weak = positive_preservation_penalty(pred, torch.full_like(pred, -1.), target)
    assert strong > weak > 0


def test_rejects_invalid_inputs():
    t = torch.zeros((1, 1, 4, 4))
    with pytest.raises(ValueError):
        positive_preservation_penalty(t, t[:, :, :2], t)
    with pytest.raises(ValueError):
        positive_preservation_penalty(t, t, t + 2)
    with pytest.raises(ValueError):
        expand_logits(t + float('nan'), 3)
