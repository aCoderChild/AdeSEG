"""Regional readouts from SAM2's already-projected, rotary-encoded attention."""
import torch
import torch.nn.functional as F
from torch import nn


def regional_readouts(query, key, value, labels, pointer_tokens, *, sdpa=None):
    """Use PyTorch SDPA on each spatial region; native pointer reading stays separate."""
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError('Expected attention tensors [batch, heads, tokens, channels]')
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError('Query/key attention dimensions differ')
    if pointer_tokens < 0 or pointer_tokens >= key.shape[-2]:
        raise ValueError('Invalid pointer count')
    spatial = key.shape[-2] - pointer_tokens
    sdpa = sdpa or F.scaled_dot_product_attention
    if labels.shape != (query.shape[0], spatial):
        raise ValueError('Labels must match spatial keys')
    if not torch.all((labels >= 0) & (labels <= 2)):
        raise ValueError('Expected background, foreground and boundary labels')
    outputs, valid = [], []
    for region in range(3):
        mask = labels.eq(region)
        present = mask.any(dim=-1)
        safe_mask = mask | (~present[:, None] & (torch.arange(spatial, device=mask.device) == 0))
        out = sdpa(query, key[..., :spatial, :], value[..., :spatial, :],
                   attn_mask=safe_mask[:, None, None, :], dropout_p=0.0)
        outputs.append(out * present[:, None, None, None])
        valid.append(present)
    return torch.stack(outputs, dim=1), torch.stack(valid, dim=1)


class RegionalReadoutResidual(nn.Module):
    """Add a zero-initialized query-dependent residual to the native read."""

    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Linear(3 * channels, channels, bias=False)
        nn.init.zeros_(self.mix.weight)

    def forward(self, native, query, key, value, labels, pointer_tokens, mode='regional', *, sdpa=None):
        if mode not in ('regional', 'generic'):
            raise ValueError('Unknown readout mode')
        if mode == 'generic':
            labels = torch.zeros_like(labels)
        reads, valid = regional_readouts(query, key, value, labels, pointer_tokens, sdpa=sdpa)
        if mode == 'generic':
            reads = reads[:, :1].expand_as(reads)
        context = reads.permute(0, 2, 3, 1, 4).flatten(-2)
        if native.shape != context.shape[:-1] + (self.mix.out_features,):
            raise ValueError('Native attention and regional readout shapes differ')
        return native + self.mix(context), valid
