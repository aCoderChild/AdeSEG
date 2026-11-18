"""Structured refiner must remain a strict-checkpoint compatible Lean adapter."""
from modeling.feature_aggregation import MultistageFeatureAggregator
from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1


def test_structured_model_inherits_legacy_runtime_contract():
    assert issubclass(MultistageFeatureAggregator, MultiscaleChannelValueRefinerV1)
    config = dict(early_channels=2, deep_channels=4, hidden=8, attention_size=4)
    original = MultiscaleChannelValueRefinerV1(**config)
    structured = MultistageFeatureAggregator(**config)
    assert list(original.state_dict()) == list(structured.state_dict())
    assert all(original.state_dict()[k].shape == structured.state_dict()[k].shape
               for k in original.state_dict())
