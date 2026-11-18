"""Parity, checkpoint-state, and gradient tests for six architectural blocks."""
import pytest
import torch

from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1
from modeling.structured_feature_aggregation import StructuredMultistageFeatureAggregator


@pytest.mark.parametrize("mode", MultiscaleChannelValueRefinerV1.MODES)
def test_structured_forward_matches_original_exactly(mode):
    torch.manual_seed(23)
    kwargs = dict(mode=mode, early_channels=2, deep_channels=4, hidden=8, attention_size=5)
    original = MultiscaleChannelValueRefinerV1(**kwargs).train()
    structured = StructuredMultistageFeatureAggregator(**kwargs).train()
    assert list(original.state_dict()) == list(structured.state_dict())
    structured.load_state_dict(original.state_dict(), strict=True)

    early = torch.randn(1, 2, 7, 9)
    deep = torch.randn(1, 4, 3, 5)
    logits = torch.randn(1, 1, 12, 14)
    old_result = original(early, deep, logits)
    new_result = structured(early, deep, logits)
    torch.testing.assert_close(old_result, new_result, atol=0, rtol=0)

    old_result.square().mean().backward()
    new_result.square().mean().backward()
    for (name, old), (other_name, new) in zip(original.named_parameters(), structured.named_parameters()):
        assert name == other_name
        if old.grad is None:
            assert new.grad is None
        else:
            torch.testing.assert_close(old.grad, new.grad, atol=0, rtol=0)


def test_runtime_type_and_untrained_inference_guards():
    structured = StructuredMultistageFeatureAggregator(
        early_channels=2, deep_channels=4, hidden=8, attention_size=4).eval()
    assert isinstance(structured, MultiscaleChannelValueRefinerV1)
    with pytest.raises(RuntimeError, match="Untrained refinement"):
        with torch.no_grad():
            structured(torch.randn(1, 2, 4, 4),
                       torch.randn(1, 4, 2, 2),
                       torch.randn(1, 1, 6, 6))
