"""Public API: six explicit blocks with historical checkpoint compatibility.

The structured model subclasses the unaltered version-1 module; all Conv2d
weights and strict source-hash checks remain registered on that parent.
Use MultistageFeatureAggregator for NEW experiments. The legacy class is
retained for exact historical reproduction and independent parity tests.
"""
from modeling.multiscale_channel_value_refiner_v1 import (
    MultiscaleChannelValueRefinerV1,
    channel_matrix,
)
from modeling.structured_feature_aggregation import StructuredMultistageFeatureAggregator

MultistageFeatureAggregator = StructuredMultistageFeatureAggregator

__all__ = [
    "MultistageFeatureAggregator",
    "StructuredMultistageFeatureAggregator",
    "MultiscaleChannelValueRefinerV1",
    "channel_matrix",
]
