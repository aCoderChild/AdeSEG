"""Reuse Cutie's learned soft pooling and attention; preserve native dense reads.

This is a project adaptation, not Cutie reproduction. No Cutie checkpoint is
transferred to a different feature space. The new module alone is trained.
"""

import sys
import torch
from torch import nn
from omegaconf import OmegaConf

from modeling.cutie_memory import author_summarizer
from modeling.cutie_region_queries import author_modules


def _author_classes():
    # Existing provenance loaders add Cutie to sys.path. Keep its package
    # available through Python's module cache without leaking its scripts/
    # namespace into subsequent project imports.
    previous = sys.path.copy()
    try:
        author_modules()
        from cutie.model.transformer.transformer_layers import CrossAttention
        return author_summarizer(), CrossAttention
    finally:
        sys.path[:] = previous


class ComplementaryMemoryReader(nn.Module):
    """Four soft summaries per frame (two object, two context), one output read.

The generic control has identical parameters and summaries, with uniform 0.5
mask cues. SAM2 object pointers remain in the native read, outside this branch.
"""

    def __init__(self, memory_dim=64, channels=256, summaries=4, heads=8):
        super().__init__()
        if summaries <= 0 or summaries % 2 or channels % heads:
            raise ValueError("Expected even summaries and channels divisible by heads")
        ObjectSummarizer, CrossAttention = _author_classes()
        self.summarizer = ObjectSummarizer(OmegaConf.create({
            "value_dim": memory_dim, "pixel_pe_scale": 32,
            "pixel_pe_temperature": 128,
            "object_summarizer": {"embed_dim": channels,
                                  "num_summaries": summaries, "add_pe": False},
        }))
        self.position_projection = nn.Linear(memory_dim, channels, bias=False)
        self.reader = CrossAttention(channels, heads, residual=False,
                                     add_pe_to_qkv=[True, True, False])
        self.output_projection = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        self.summaries = summaries

    def summarize(self, memory, positions, probabilities, mode):
        # B = separate objects; T = selected native memory frames. Never mix B.
        if mode not in ("regional", "generic"):
            raise ValueError("Expected regional or generic control")
        if memory.ndim != 5 or positions.shape != memory.shape:
            raise ValueError("Expected memory and positions [B,T,C,H,W]")
        b, t, c, h, w = memory.shape
        if t == 0 or probabilities.shape != (b, t, h, w):
            raise ValueError("Stored probability maps must match spatial memory")
        if (not torch.isfinite(memory).all() or not torch.isfinite(positions).all()
                or not torch.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()):
            raise ValueError("Expected finite features and probabilities in [0,1]")
        masks = probabilities.float() if mode == "regional" else torch.full_like(probabilities, .5)
        # Cutie's object axis is used here for frames of the SAME object.
        # Its pooling is independent along that axis; no object IDs are pooled.
        summaries, logits = self.summarizer(masks, memory.float(), need_weights=True)
        half = self.summaries // 2
        memberships = torch.cat((masks[..., None].expand(-1, -1, -1, -1, half),
                                 (1 - masks[..., None]).expand(-1, -1, -1, -1, half)), -1)
        from cutie.model.transformer.object_summarizer import _weighted_pooling
        # Pool native spatial+temporal encodings with the EXACT feature weights.
        position_sums, area = _weighted_pooling(
            memberships, positions.float().permute(0, 1, 3, 4, 2), logits)
        # Numerical denominator protection, not a confidence decision. Matches
        # the existing Cutie summary normalization; no groups are discarded.
        denominator = area.clamp_min(1e-6)
        messages = summaries[..., :-1] / denominator
        pos = self.position_projection(position_sums / denominator)
        # Empty groups remain zero, with no confidence gate or token deletion.
        return messages.flatten(1, 2), pos.flatten(1, 2)

    def forward(self, native, query_positions, memory, positions, probabilities, mode):
        if native.ndim != 3 or query_positions.shape != native.shape:
            raise ValueError("Expected batch-first native output and positions [B,Q,C]")
        if native.shape[0] != memory.shape[0]:
            raise ValueError("Memory and query object batches differ")
        messages, message_positions = self.summarize(memory, positions, probabilities, mode)
        branch, _ = self.reader(native.float(), messages, query_positions.float(), message_positions)
        update = self.output_projection(branch)
        if not torch.isfinite(update).all():
            raise RuntimeError("Nonfinite complementary residual")
        return native + update.to(native.dtype)
