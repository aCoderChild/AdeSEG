"""Bounded region gains on SAM2 memory before key/value projections."""
import torch
from torch import nn


class RegionKeyValueGate(nn.Module):
    def __init__(self, channels=256, max_gain_change=0.25):
        super().__init__()
        self.gain_logits = nn.Parameter(torch.zeros(3, channels))
        self.max_gain_change = float(max_gain_change)

    def forward(self, memory, labels, pointer_tokens):
        # MemoryAttention uses batch_first=True at each cloned layer, so
        # layer.forward receives [batch=1, tokens, channels].
        if memory.ndim != 3 or memory.shape[0] != 1:
            raise ValueError('Expected layer memory [1, tokens, channels]')
        if not 0 <= pointer_tokens < memory.shape[1]:
            raise ValueError('Invalid pointer count')
        spatial = memory.shape[1] - pointer_tokens
        if labels.shape != (1, spatial) or not torch.all((labels >= 0) & (labels <= 2)):
            raise ValueError('Region labels must match spatial memory tokens')
        if memory.shape[-1] != self.gain_logits.shape[-1]:
            raise ValueError('Gate channels do not match SAM2 memory')
        gain = 1.0 + self.max_gain_change * self.gain_logits.tanh()
        per_token = gain[labels[0]].to(memory.dtype)
        spatial_memory = memory[:, :spatial] * per_token[None, :, :]
        return torch.cat((spatial_memory, memory[:, spatial:]), dim=1)


class RegionKeyValueGateHeads(nn.Module):
    def __init__(self, layers=4, channels=256, max_gain_change=0.25):
        super().__init__()
        self.layers = nn.ModuleList(
            RegionKeyValueGate(channels, max_gain_change) for _ in range(layers))
