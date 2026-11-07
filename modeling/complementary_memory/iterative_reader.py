"""Iterative current-pixel/memory interaction using unchanged Cutie blocks.

This adapts author components to frozen SAM2, not a reproduction of Cutie.
Cutie's hard auxiliary attention masks and mask-prediction heads are not used.
"""
import math
import sys

import torch
from torch import nn
from omegaconf import OmegaConf

from modeling.complementary_memory.reader import ComplementaryMemoryReader
from modeling.cutie_region_queries import author_modules


class IterativeMemoryReader(ComplementaryMemoryReader):
    """Soft native-memory summaries interact with current pixels three times.

    The generic control replaces occupancy with 0.5 in the inherited pooling.
    Both controls retain every native spatial and pointer token, and have the
    same trainable parameters. Separate objects occupy separate batch entries.
    """

    def __init__(self, memory_dim=64, channels=256, summaries=16, heads=8,
                 blocks=3, feedforward=2048):
        super().__init__(memory_dim, channels, summaries, heads)
        if blocks < 1:
            raise ValueError("At least one author interaction block is required")
        previous = sys.path.copy()
        try:
            block_class, _ = author_modules()
        finally:
            sys.path[:] = previous
        config = OmegaConf.create({"object_transformer": {
            "embed_dim": channels, "num_heads": heads, "num_queries": summaries,
            "ff_dim": feedforward,
            "read_from_pixel": {"add_pe_to_qkv": [True, True, False]},
            "query_self_attention": {"add_pe_to_qkv": [True, True, False]},
            "read_from_query": {"add_pe_to_qkv": [True, True, False], "output_norm": False},
        }})
        # Replace the inherited single cross-attention module; no unused parameters.
        self.reader = nn.ModuleList(block_class(config) for _ in range(blocks))

    def forward(self, native, query_positions, memory, positions, probabilities, mode):
        if native.ndim != 3 or query_positions.shape != native.shape:
            raise ValueError("Expected [B,Q,C] native current features and positions")
        b, q, c = native.shape
        side = math.isqrt(q)
        if side*side != q or b != memory.shape[0]:
            raise ValueError("Expected square current grid and object-aligned batches")
        query, query_pe = self.summarize(memory, positions, probabilities, mode)
        pixel = native.float().transpose(1,2).reshape(b,1,c,side,side)
        initial = pixel
        for block in self.reader:
            query, pixel, _, _ = block(query, pixel, query_pe, query_positions.float(),
                                       attn_mask=None, need_weights=False)
        delta = (pixel-initial).flatten(3).squeeze(1).transpose(1,2)
        update = self.output_projection(delta)
        if not torch.isfinite(update).all():
            raise RuntimeError("Nonfinite iterative memory update")
        return native + update.to(native.dtype)
