"""Apply a regional residual inside SAM2's native memory cross-attention."""
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from modeling.cfbi_region_gate import NativeCFBIGate
from modeling.region_readout import RegionalReadoutResidual
from modeling.region_mass_attention import RegionMassBias


class _Labels(nn.Module):
    def region_context(self, vectors, labels, mode):
        self.labels = labels[None]
        return vectors.new_zeros((1, 192))

    def forward(self, attention, context):
        return attention


class RegionalReadoutHeads(nn.Module):
    def __init__(self, channels=256, layers=4):
        super().__init__()
        self.layers = nn.ModuleList(RegionalReadoutResidual(channels) for _ in range(layers))


class RegionalMassHeads(nn.Module):
    def __init__(self, layers=4):
        super().__init__()
        self.layers = nn.ModuleList(RegionMassBias() for _ in range(layers))


class NativeRegionalReadout:
    def __init__(self, model, heads, mode, sequence, shuffle_seed=20261031,
                 region_encoder=None):
        if mode not in ('regional', 'shuffled', 'generic', 'regional_value', 'regional_gain',
                        'soft_role_attention', 'soft_role_global'):
            raise ValueError('Unknown control')
        self.model, self.heads, self.mode = model, heads, mode
        self.labels = _Labels()
        observer_mode = 'shuffled' if mode == 'shuffled' else 'regional'
        self.observer = NativeCFBIGate(model, self.labels, observer_mode, sequence,
                                       shuffle_seed, region_encoder=region_encoder)
        self.active = None
        self.in_adapter = False
        self.calls = []
        self.query_content = {}

    def _before(self, layer):
        def before(module, args, kwargs):
            if self.active is not None:
                raise RuntimeError('Nested memory attention')
            if getattr(self.heads, 'requires_query_content', False):
                q_input = kwargs.get('q', args[0] if args else None)
                if q_input is None:
                    raise RuntimeError('Memory attention query input is missing')
                self.query_content[layer] = module._separate_heads(
                    module.q_proj(q_input), module.num_heads).detach()
            self.active = (layer, int(kwargs.get('num_k_exclude_rope', 0)))
        return before

    def _after(self, module, args, kwargs, output):
        self.active = None

    def _attention(self, q, k, v, *args, **kwargs):
        if self.in_adapter:
            return self.sdpa(q, k, v, *args, **kwargs)
        native = self.sdpa(q, k, v, *args, **kwargs)
        if self.active is None:
            return native
        if args or kwargs.get('attn_mask') is not None or kwargs.get('dropout_p', 0) != 0:
            raise RuntimeError('Unsupported memory attention parameters')
        layer, pointers = self.active
        self.in_adapter = True
        try:
            if self.mode in ('regional_value', 'regional_gain'):
                head_mode = self.mode
            elif self.mode == 'soft_role_attention':
                head_mode = 'regional'
            elif self.mode == 'soft_role_global':
                head_mode = 'global'
            else:
                head_mode = 'generic' if self.mode == 'generic' else 'regional'
            output, valid = self.heads.layers[layer](native, q, k, v, self.labels.labels, pointers,
                                                     head_mode,
                                                     **({'query_content': self.query_content[layer]}
                                                        if getattr(self.heads, 'requires_query_content', False) else {}),
                                                     sdpa=self.sdpa)
        finally:
            self.in_adapter = False
        self.calls.append({'layer': layer, 'pointer_keys': pointers, 'spatial_keys': k.shape[-2] - pointers,
                           'valid_regions': valid.detach().cpu().tolist()})
        return output

    def __enter__(self):
        self.observer.__enter__()
        self.handles = []
        for i, layer in enumerate(self.model.memory_attention.layers):
            attention = layer.cross_attn_image
            self.handles.append(attention.register_forward_pre_hook(self._before(i), with_kwargs=True))
            self.handles.append(attention.register_forward_hook(self._after, with_kwargs=True, always_call=True))
        self.sdpa = F.scaled_dot_product_attention
        self.override = patch.object(F, 'scaled_dot_product_attention', self._attention)
        self.override.start()
        return self

    def __exit__(self, *exc):
        self.override.stop()
        for handle in self.handles:
            handle.remove()
        self.observer.__exit__(*exc)
