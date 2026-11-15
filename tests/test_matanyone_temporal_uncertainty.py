import torch

from modeling.matanyone_temporal_uncertainty import (
    MATANYONE_REVISION,
    author_uncertainty_class,
    build_temporal_uncertainty,
)


def test_temporal_uncertainty_uses_pinned_author_class():
    assert MATANYONE_REVISION == "e5ddc534c1fff9bb9e54cf476095d29071b7cb4f"
    assert author_uncertainty_class().__name__ == "UncertPred"


def test_temporal_uncertainty_forward_and_gradients():
    torch.manual_seed(11)
    module = build_temporal_uncertainty().train()
    last = torch.randn(1, 256, 8, 8, requires_grad=True)
    current = torch.randn(1, 256, 8, 8, requires_grad=True)
    mask = torch.rand(1, 1, 8, 8)
    difference = torch.randn(1, 256, 8, 8, requires_grad=True)

    logits = module(last, current, mask, difference)
    assert logits.shape == (1, 1, 8, 8)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in module.parameters())
    assert all(x.grad is not None and torch.isfinite(x.grad).all()
               for x in (last, current, difference))
