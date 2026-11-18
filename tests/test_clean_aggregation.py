"""Smoke-test the retained learned module and teacher preservation together."""
import torch
from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1
from modeling.training_foreground_preservation_v1 import positive_preservation_penalty


def test_multiscale_identity_and_teacher_gradients():
    torch.manual_seed(3)
    model = MultiscaleChannelValueRefinerV1(early_channels=2, deep_channels=4, hidden=8, attention_size=4)
    early = torch.randn(1, 2, 8, 8)
    deep = torch.randn(1, 4, 4, 4)
    logits = torch.randn(1, 1, 16, 16)
    target = (torch.rand_like(logits) > .5).float()
    result = model(early, deep, logits)
    assert torch.equal(result, logits), "Zero-initialized model must preserve native logits"
    loss = positive_preservation_penalty(result, logits + 0.5, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head[-1].weight.grad is not None
    assert torch.isfinite(model.head[-1].weight.grad).all()
