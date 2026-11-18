"""Six-stage Lean-SAM2 learned Q/K/V channel-value feature aggregation.

Imports the SAME registered parameters and strict source-hash verifier as the
historical MultiscaleChannelValueRefinerV1 (via subclassing).
"""
from modeling.feature_aggregation import (
    MultistageFeatureAggregator,
    StructuredMultistageFeatureAggregator,
    MultiscaleChannelValueRefinerV1,
)
from modeling.aggregation_blocks import (
    FeatureProjectionBlock,
    SpatialContextBlock,
    RegionAwareChannelAttentionBlock,
    ForegroundBackgroundContrastBlock,
    FeatureFusionBlock,
    ResidualMaskDecoderBlock,
)

__all__ = [
    "MultistageFeatureAggregator",
    "StructuredMultistageFeatureAggregator",
    "MultiscaleChannelValueRefinerV1",
    "FeatureProjectionBlock",
    "SpatialContextBlock",
    "RegionAwareChannelAttentionBlock",
    "ForegroundBackgroundContrastBlock",
    "FeatureFusionBlock",
    "ResidualMaskDecoderBlock",
]
