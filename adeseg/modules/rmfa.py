"""RMFA: stable public interface for Lean-SAM2 region-aware residual refinement.

This creates no second parameterized implementation. The exact source-bound
StructuredMultistageFeatureAggregator owns the learnable weights and can
be routed using the existing source-verified Lean inference integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from modeling.structured_feature_aggregation import StructuredMultistageFeatureAggregator
from modeling.lean_channel_value_routing_v1 import route_channel_value_refinement


@dataclass(frozen=True)
class RMFAConfig:
    mode: str = "foreground_background_asinh"
    early_channels: int = 32
    deep_channels: int = 256
    hidden_channels: int = 64
    attention_size: int = 32

    def __post_init__(self):
        if self.mode not in StructuredMultistageFeatureAggregator.MODES:
            raise ValueError("Unknown RMFA attention/contrast mode")
        if (type(self.early_channels) is not int or self.early_channels < 1
                or type(self.deep_channels) is not int or self.deep_channels < 1
                or type(self.hidden_channels) is not int or self.hidden_channels < 2
                or self.hidden_channels % 2
                or type(self.attention_size) is not int or self.attention_size < 1):
            raise ValueError("Invalid RMFA channel/attention geometry")


RMFA = StructuredMultistageFeatureAggregator


def build_rmfa(config: RMFAConfig | None = None) -> RMFA:
    config = config if config is not None else RMFAConfig()
    if not isinstance(config, RMFAConfig):
        raise TypeError("build_rmfa expects RMFAConfig")
    return RMFA(
        mode=config.mode, early_channels=config.early_channels,
        deep_channels=config.deep_channels,
        hidden=config.hidden_channels, attention_size=config.attention_size,
    )


__all__ = ["RMFA", "RMFAConfig", "build_rmfa", "route_channel_value_refinement"]
