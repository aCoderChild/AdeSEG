"""TTMR: target-aware temporal selection under an explicit token budget.

The selector currently produces candidate indices in a reference/offline
setting. A live Lean-SAM2 memory adapter requires a separate verified hook.
"""
from __future__ import annotations

from .selection import (
    TTMRConfig, rank_temporal_memory, rank_spatial_memory,
)


class TTMR:
    def __init__(self, config: TTMRConfig):
        if not isinstance(config, TTMRConfig):
            raise TypeError("TTMR expects an explicit fixed-budget TTMRConfig")
        self.config = config

    def select(self, memory_tokens, anchors, usage_scores, source_frames, *,
               current_frame, multiscale_scores=None):
        return rank_temporal_memory(
            memory_tokens, anchors, usage_scores, source_frames,
            current_frame=current_frame, config=self.config,
            multiscale_scores=multiscale_scores,
        )

    def select_spatial(self, memory_maps, anchors, usage_scores, frame_ids, *,
                       current_frame, scales=(2, 4), context_weight=0.5):
        """Compute pooled context for original memory tokens (shadow mode)."""
        return rank_spatial_memory(
            memory_maps, anchors, usage_scores, frame_ids,
            current_frame=current_frame, config=self.config,
            scales=scales, context_weight=context_weight,
        )


__all__ = ["TTMR", "TTMRConfig", "rank_temporal_memory", "rank_spatial_memory"]
