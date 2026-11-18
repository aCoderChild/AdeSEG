"""Auditable, stateless stages of multi-stage segmentation refinement.

The trainable Conv2d parameters remain registered on the legacy model so
historical state_dict keys and strict checkpoint validation are unchanged.
"""
from .projection import FeatureProjectionBlock, SpatialContextBlock
from .attention import RegionAwareChannelAttentionBlock
from .contrast import ForegroundBackgroundContrastBlock
from .fusion import FeatureFusionBlock, ResidualMaskDecoderBlock

__all__ = ["FeatureProjectionBlock","SpatialContextBlock",
           "RegionAwareChannelAttentionBlock","ForegroundBackgroundContrastBlock",
           "FeatureFusionBlock","ResidualMaskDecoderBlock"]
