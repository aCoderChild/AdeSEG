"""Bounded role embeddings for SAM2's spatial memory values."""
import torch
from torch import nn


class RegionValueAdapter(nn.Module):
    def __init__(self, heads=1, head_dim=256, max_relative=0.1):
        super().__init__()
        self.role_embeddings = nn.Parameter(torch.zeros(heads, 3, head_dim))
        self.max_relative = float(max_relative)

    def forward(self, native, query, key, value, labels, pointer_tokens,
                mode='regional_value', *, query_content=None, sdpa=None):
        if mode != 'regional_value':
            raise ValueError('Expected regional_value mode')
        if value.ndim != 4 or key.shape != value.shape or query.shape[:2] != value.shape[:2]:
            raise ValueError('Expected compatible [batch, heads, tokens, channels] tensors')
        if not 0 <= pointer_tokens < value.shape[-2]:
            raise ValueError('Invalid pointer count')
        spatial = value.shape[-2] - pointer_tokens
        if labels.shape != (value.shape[0], spatial) or not torch.all((labels >= 0) & (labels <= 2)):
            raise ValueError('Region labels must match spatial memory keys')
        if self.role_embeddings.shape[0] != value.shape[1] or self.role_embeddings.shape[2] != value.shape[-1]:
            raise ValueError('Adapter dimensions do not match memory values')
        values = value[..., :spatial, :]
        scale = values.float().square().mean(dim=-2, keepdim=True).sqrt().to(value.dtype)
        embeddings = self.max_relative * self.role_embeddings.tanh()
        embeddings = embeddings.unsqueeze(0).expand(value.shape[0], -1, -1, -1)
        index = labels[:, None, :, None].expand(-1, value.shape[1], -1, value.shape[-1])
        delta = torch.gather(embeddings, 2, index) * scale
        adapted = torch.cat((values + delta, value[..., spatial:, :]), dim=-2)
        if sdpa is None:
            raise ValueError('A scaled dot-product attention function is required')
        output = sdpa(query, key, adapted, dropout_p=0.0)
        return output, torch.ones((value.shape[0], 3), dtype=torch.bool, device=value.device)


class RegionValueHeads(nn.Module):
    def __init__(self, layers=4, heads=1, head_dim=256):
        super().__init__()
        self.layers = nn.ModuleList(
            RegionValueAdapter(heads, head_dim) for _ in range(layers))
