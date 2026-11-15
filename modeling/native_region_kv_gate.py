"""Apply predicted-region gains before SAM2's memory key/value projections."""
import torch

from modeling.cfbi_region_gate import NativeCFBIGate
from modeling.native_region_readout import _Labels


class NativeRegionalKVGate:
    def __init__(self, model, heads, sequence):
        self.model, self.heads, self.sequence = model, heads, sequence
        self.labels = _Labels()
        self.observer = NativeCFBIGate(model, self.labels, 'regional', sequence)
        self.calls = []
        self.handles = []

    def _memory(self, layer):
        def before(module, args, kwargs):
            if args or 'memory' not in kwargs:
                raise RuntimeError('Unexpected SAM2 memory-attention layer signature')
            memory = kwargs['memory']
            pointers = int(kwargs.get('num_k_exclude_rope', 0))
            labels = self.labels.labels
            if memory.shape[0] != 1:
                raise RuntimeError(f'Expected batch-first SAM2 layer memory, got {tuple(memory.shape)}')
            spatial = memory.shape[1] - pointers
            if spatial != labels.shape[1] or pointers < 0:
                raise RuntimeError('SAM2 spatial/pointer key counts do not match stored-mask labels')
            adapted = self.heads.layers[layer](memory, labels, pointers)
            pointers_unchanged = pointers == 0 or torch.equal(
                adapted[:, -pointers:], memory[:, -pointers:])
            if not pointers_unchanged:
                raise RuntimeError('Regional gate modified object-pointer tokens')
            self.calls.append({'layer': layer, 'spatial_keys': spatial,
                               'pointer_keys': pointers,
                               'pointer_tokens_unchanged': pointers_unchanged,
                               'region_counts': [(labels == region).sum(dim=1).tolist()[0]
                                                 for region in range(3)]})
            self.observer.trace[-1]['layers'].append(layer)
            return args, {**kwargs, 'memory': adapted}
        return before

    def __enter__(self):
        self.observer.__enter__()
        for index, layer in enumerate(self.model.memory_attention.layers):
            self.handles.append(layer.register_forward_pre_hook(
                self._memory(index), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.observer.__exit__(*exc)
