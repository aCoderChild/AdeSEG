"""Scientifically named, checkpoint-compatible public API for Lean-SAM2 aggregation.

This module re-exports the original class object, not a duplicated class.
Video routing and provenance checks still require the exact original class
and the SHA256 of modeling/multiscale_channel_value_refiner_v1.py.
"""
from modeling.multiscale_channel_value_refiner_v1 import (
    MultiscaleChannelValueRefinerV1,
    channel_matrix,
)

MultistageFeatureAggregator = MultiscaleChannelValueRefinerV1

__all__ = ["MultistageFeatureAggregator", "MultiscaleChannelValueRefinerV1", "channel_matrix"]
