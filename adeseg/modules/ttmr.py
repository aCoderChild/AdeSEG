"""TTMR: target-aware temporal selection under an explicit token budget.

The selector currently produces candidate indices in a reference/offline
setting. A live Lean-SAM2 memory adapter requires a separate verified hook.
"""
from __future__ import annotations

from adeseg.memory.target_aware_temporal import (
    TTMRConfig, rank_temporal_memory,
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


__all__ = ["TTMR", "TTMRConfig", "rank_temporal_memory"]
