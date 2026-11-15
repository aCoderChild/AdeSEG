"""Query-conditioned regional attention bias for SAM2 memory reads."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from modeling.region_mass_attention import redistribute_region_mass


class QueryConditionedRegionReadout(nn.Module):
    def __init__(self, heads=1, head_dim=256, max_log_bias=math.log(2.0)):
        super().__init__()
        self.query_gate = nn.Parameter(torch.zeros(heads, head_dim, 3))
        self.max_log_bias = float(max_log_bias)

    def forward(self, native, query, key, value, labels, pointer_tokens,
                mode='regional', *, query_content=None, sdpa=None):
        if mode != 'regional':
            raise ValueError('Expected regional mode')
        if query_content is None or query_content.shape != query.shape:
            raise ValueError('Expected projected query features matching attention query shape')
        if query.shape[1] != self.query_gate.shape[0] or query.shape[-1] != self.query_gate.shape[1]:
            raise ValueError('Attention head dimensions do not match the query gate')

        query_content = F.normalize(query_content.float(), dim=-1)
        scores = torch.einsum('bhqd,hdr->bhqr', query_content, self.query_gate.float())
        log_bias = self.max_log_bias * scores.tanh()
        return redistribute_region_mass(query, key, value, labels, pointer_tokens,
                                         log_bias, sdpa=sdpa)


class QueryConditionedRegionHeads(nn.Module):
    requires_query_content = True

    def __init__(self, layers=4, heads=1, head_dim=256):
        super().__init__()
        self.layers = nn.ModuleList(
            QueryConditionedRegionReadout(heads, head_dim) for _ in range(layers))
