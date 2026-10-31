"""CFBI instance-attention gate conditioned on SAM2 predicted-region memories."""
import ast
import hashlib
import inspect
from functools import lru_cache
from unittest.mock import patch

import torch
from torch import nn

from modeling.region_bias import feature_key
from modeling.region_definitions import mixed_cell_regions
from modeling.region_tokens import _verified_source
from modeling.regional_refinement import pooled_context

CFBI_REVISION = '91fc6a31b1752a8a9110b3ed480e05656f756691'


@lru_cache(maxsize=1)
def author_gate():
    path, source = _verified_source('CFBI', CFBI_REVISION, 'networks/layers/attention.py')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'IA_gate')
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['IA_gate']


class RegionGate(nn.Module):
    """One shared author gate; zero initialization exactly preserves native output."""
    def __init__(self):
        super().__init__()
        self.gate = author_gate()(192, 256)
        nn.init.zeros_(self.gate.IA.weight)
        nn.init.zeros_(self.gate.IA.bias)

    def forward(self, attention, context):
        if attention.ndim != 3 or attention.shape[-1] != 256 or context.shape != (attention.shape[0], 192):
            raise ValueError('Expected attention [B,Q,256] and context [B,192]')
        return self.gate(attention.transpose(1, 2).unsqueeze(-1), context).squeeze(-1).transpose(1, 2)


def region_context(vectors, labels, mode, seed=0):
    if vectors.ndim != 2 or vectors.shape[1] != 64 or labels.shape != (len(vectors),):
        raise ValueError('Expected spatial memory [N,64] and aligned region labels')
    if mode not in ('regional', 'shuffled', 'generic'):
        raise ValueError('Unknown control')
    if mode == 'shuffled':
        permutation = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed)).to(labels.device)
        labels = labels[permutation]
    if mode == 'generic':
        return vectors.mean(0).repeat(3).unsqueeze(0)
    context, _, _ = pooled_context(vectors, labels)
    return context.squeeze(-1).T.reshape(1, 192)


class NativeCFBIGate:
    """Attach the shared gate to the four native cross-attention outputs."""
    def __init__(self, model, gate, mode, sequence):
        self.model, self.gate, self.mode, self.sequence = model, gate, mode, sequence
        self.prepare = model._prepare_memory_conditioned_features
        self.signature = inspect.signature(self.prepare)
        self.handles = []
        self.trace = []
        self.context = None

    def prepare_read(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        data = bound.arguments
        frame = data['frame_idx']
        self.frame, self.bank, self.context = frame, {}, None
        if not data['is_init_cond_frame']:
            for outputs in data['output_dict'].values():
                for index, stored in outputs.items():
                    if index >= frame or stored.get('maskmem_features') is None:
                        continue
                    key = feature_key(stored['maskmem_features'].flatten(2).transpose(1, 2))
                    if key in self.bank:
                        raise RuntimeError('Ambiguous native memory identity')
                    labels = torch.from_numpy(mixed_cell_regions(stored['pred_masks'][0, 0])['labels']).flatten()
                    self.bank[key] = (index, labels)
        return self.prepare(*args, **kwargs)

    def read(self, module, args, kwargs):
        if args:
            raise RuntimeError('Unexpected memory-attention signature')
        memory = kwargs['memory']
        pointers = int(kwargs.get('num_obj_ptr_tokens', 0))
        spatial = len(memory) - pointers
        if memory.shape[1] != 1 or spatial <= 0 or spatial % 4096:
            raise RuntimeError('Unsupported memory layout')
        labels, frames = [], []
        for start in range(0, spatial, 4096):
            key = feature_key(memory[start:start+4096].transpose(0, 1))
            if key not in self.bank:
                raise RuntimeError('Memory does not match a stored predicted mask')
            index, label = self.bank[key]
            if index >= self.frame:
                raise RuntimeError('Future memory access')
            frames.append(index)
            if self.mode == 'shuffled':
                seed = int.from_bytes(hashlib.sha256(f'{self.sequence}/{index}/20261031'.encode()).digest()[:8], 'little') % (2**63)
                label = label[torch.randperm(len(label), generator=torch.Generator().manual_seed(seed))]
            labels.append(label)
        vectors = memory[:spatial, 0].float()
        mode = 'regional' if self.mode == 'shuffled' else self.mode
        self.context = region_context(vectors, torch.cat(labels).to(vectors.device), mode)
        self.trace.append({'frame': self.frame, 'memory_frames': frames, 'spatial_keys': spatial, 'pointer_keys': pointers, 'layers': []})

    def apply(self, layer, module, args, output):
        if self.context is None or not self.trace:
            raise RuntimeError('Missing memory region context')
        self.trace[-1]['layers'].append(layer)
        return self.gate(output, self.context.to(device=output.device, dtype=output.dtype))

    def __enter__(self):
        self.override = patch.object(self.model, '_prepare_memory_conditioned_features', self.prepare_read)
        self.override.start()
        self.handles.append(self.model.memory_attention.register_forward_pre_hook(self.read, with_kwargs=True))
        for i, layer in enumerate(self.model.memory_attention.layers):
            self.handles.append(layer.cross_attn_image.register_forward_hook(
                lambda module, args, output, i=i: self.apply(i, module, args, output)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.override.stop()
