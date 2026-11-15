"""Learned region-conditioned multiplicative gates for spatial memory values."""
import torch
from torch import nn


class RegionValueGate(nn.Module):
    def __init__(self, heads=1, head_dim=256, max_gain_change=0.25):
        super().__init__()
        self.gain_logits = nn.Parameter(torch.zeros(heads, 3, head_dim))
        self.max_gain_change = float(max_gain_change)

    def forward(self, native, query, key, value, labels, pointer_tokens,
                mode='regional_gain', *, query_content=None, sdpa=None):
        if mode != 'regional_gain':
            raise ValueError('Expected regional_gain mode')
        if value.ndim != 4 or key.shape != value.shape or query.shape[:2] != value.shape[:2]:
            raise ValueError('Expected compatible [batch, heads, tokens, channels] tensors')
        if not 0 <= pointer_tokens < value.shape[-2]:
            raise ValueError('Invalid pointer count')
        spatial = value.shape[-2] - pointer_tokens
        if labels.shape != (value.shape[0], spatial) or not torch.all((labels >= 0) & (labels <= 2)):
            raise ValueError('Region labels must match spatial memory keys')
        if self.gain_logits.shape[0] != value.shape[1] or self.gain_logits.shape[2] != value.shape[-1]:
            raise ValueError('Gate dimensions do not match memory values')
        index = labels[:, None, :, None].expand(-1, value.shape[1], -1, value.shape[-1])
        logits = self.gain_logits.unsqueeze(0).expand(value.shape[0], -1, -1, -1)
        relative_gain = 1.0 + self.max_gain_change * torch.tanh(torch.gather(logits, 2, index))
        adapted = torch.cat((value[..., :spatial, :] * relative_gain,
                             value[..., spatial:, :]), dim=-2)
        if sdpa is None:
            raise ValueError('A scaled dot-product attention function is required')
        output = sdpa(query, key, adapted, dropout_p=0.0)
        return output, torch.ones((value.shape[0], 3), dtype=torch.bool, device=value.device)


class RegionValueGateHeads(nn.Module):
    def __init__(self, layers=4, heads=1, head_dim=256, max_gain_change=0.25):
        super().__init__()
        self.layers = nn.ModuleList(
            RegionValueGate(heads, head_dim, max_gain_change) for _ in range(layers))
