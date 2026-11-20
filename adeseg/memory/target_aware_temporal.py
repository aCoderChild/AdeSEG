"""Compatibility import for TTMR's causal memory selector.

Canonical implementation: adeseg.modules.ttmr.selection.
"""
from adeseg.modules.ttmr.selection import TTMRConfig, rank_temporal_memory

__all__ = ["TTMRConfig", "rank_temporal_memory"]
