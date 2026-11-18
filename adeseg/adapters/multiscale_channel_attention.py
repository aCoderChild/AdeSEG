"""Learned Lean-SAM2 multi-stage feature/channel-value aggregation.

This class subclasses the exact source-bound legacy model so its registered
parameters and checkpoint source digest remain compatible.
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
