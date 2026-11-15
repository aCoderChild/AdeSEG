"""Low-dimensional reweighting of SAM2 spatial memory attention by mask region."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from modeling.region_readout import regional_readouts


def redistribute_region_mass(query, key, value, labels, pointer_tokens, log_bias, *, sdpa=None):
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError('Expected attention tensors [batch, heads, tokens, channels]')
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError('Query/key attention dimensions differ')
    query_bias_shape = (*query.shape[:-1], 3)
    if log_bias.shape not in ((3,), query_bias_shape) or pointer_tokens < 0 or pointer_tokens >= key.shape[-2]:
        raise ValueError('Invalid regional bias or pointer count')

    spatial = key.shape[-2] - pointer_tokens
    if labels.shape != (query.shape[0], spatial):
        raise ValueError('Labels must match spatial keys')
    if not torch.all((labels >= 0) & (labels <= 2)):
        raise ValueError('Expected background, foreground and boundary labels')

    sdpa = sdpa or F.scaled_dot_product_attention
    regional_outputs, valid = regional_readouts(query, key, value, labels, pointer_tokens, sdpa=sdpa)
    logits = torch.matmul(query.float(), key[..., :spatial, :].float().transpose(-2, -1))
    logits = logits / math.sqrt(query.shape[-1])

    log_masses = []
    for region in range(3):
        region_mask = labels.eq(region)
        present = region_mask.any(-1)
        safe_mask = region_mask | (~present[:, None] & (torch.arange(spatial, device=labels.device) == 0))
        regional_logits = logits.masked_fill(~safe_mask[:, None, None, :], float('-inf'))
        log_mass = torch.logsumexp(regional_logits, dim=-1)
        log_masses.append(log_mass.masked_fill(~present[:, None, None], float('-inf')))
    log_masses = torch.stack(log_masses, dim=-1)
    region_weights = torch.softmax(log_masses + log_bias.float(), dim=-1)
    regional_outputs = regional_outputs.permute(0, 2, 3, 1, 4).float()
    regional = (regional_outputs * region_weights[..., None]).sum(dim=3)
    log_spatial_mass = torch.logsumexp(log_masses, dim=-1)

    if pointer_tokens:
        pointer_key = key[..., spatial:, :]
        pointer_value = value[..., spatial:, :]
        pointer_logits = torch.matmul(query.float(), pointer_key.float().transpose(-2, -1))
        pointer_logits = pointer_logits / math.sqrt(query.shape[-1])
        log_pointer_mass = torch.logsumexp(pointer_logits, dim=-1)
        total = torch.logsumexp(torch.stack((log_spatial_mass, log_pointer_mass), dim=-1), dim=-1)
        spatial_weight = torch.exp(log_spatial_mass - total)
        pointer_weight = torch.exp(log_pointer_mass - total)
        pointer = sdpa(query, pointer_key, pointer_value, dropout_p=0.0).float()
        output = spatial_weight[..., None] * regional + pointer_weight[..., None] * pointer
    else:
        output = regional

    output = output.to(dtype=query.dtype)
    if torch.count_nonzero(log_bias.detach()).item() == 0:
        output = native_identity(query, key, value, sdpa, output)
    return output, valid


def native_identity(query, key, value, sdpa, computed):
    """Keep an exact native forward at zero bias while retaining bias gradients."""
    native = sdpa(query, key, value, dropout_p=0.0)
    if torch.is_grad_enabled():
        return _ExactNativeForward.apply(native, computed)
    return native


class _ExactNativeForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native, computed):
        return native

    @staticmethod
    def backward(ctx, gradient):
        return torch.zeros_like(gradient), gradient


class RegionMassBias(nn.Module):
    """Three learned regional log-mass biases; one small parameter set per layer."""

    def __init__(self, max_log_scale=math.log(2.0)):
        super().__init__()
        self.log_bias = nn.Parameter(torch.zeros(3))
        self.max_log_scale = float(max_log_scale)

    def forward(self, native, query, key, value, labels, pointer_tokens, mode='regional', *, sdpa=None):
        if mode == 'generic':
            valid = torch.stack([labels.eq(i).any(-1) for i in range(3)], dim=-1)
            return native, valid
        if mode != 'regional':
            raise ValueError('Unknown readout mode')
        bias = self.max_log_scale * torch.tanh(self.log_bias)
        return redistribute_region_mass(query, key, value, labels, pointer_tokens, bias, sdpa=sdpa)
