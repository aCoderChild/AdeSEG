"""Apply CFBI regional matching evidence inside the native SAM2 read."""

from unittest.mock import patch

import torch

from modeling.cfbi_region_affinity import regional_matching_evidence
from modeling.cfbi_region_gate import NativeCFBIGate


class _IdentityRegionContext(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None

    def region_context(self, vectors, labels, mode):
        if mode != "regional":
            raise ValueError("Expected predicted-mask regional labels")
        self.labels = labels[None]
        return vectors.new_zeros((1, 192))

    def forward(self, attention, context):
        return attention


class NativeCFBIRegionAffinity:
    def __init__(self, model, heads, sequence, *, chunks=32):
        self.model = model
        self.heads = heads
        self.sequence = sequence
        self.chunks = chunks
        self.identity = _IdentityRegionContext()
        self.observer = NativeCFBIGate(model, self.identity, "regional", sequence, 0)
        self.pending = {}
        self.calls = []

    @staticmethod
    def _feature_map(value):
        batch, channels, tokens = value.shape
        if batch != 1 or tokens != 64 * 64:
            raise RuntimeError("Expected one 64x64 cross-attention query")
        return value.reshape(batch, channels, 64, 64)

    @staticmethod
    def _attention_map(value):
        if value.ndim != 4 or value.shape != (1, 256, 64, 64):
            raise RuntimeError("Regional adapter changed SAM2 attention geometry")
        return value.flatten(2).transpose(1, 2).transpose(0, 1)

    def _before(self, layer):
        def before(module, args, kwargs):
            if args:
                raise RuntimeError("Unexpected SAM2 attention arguments")
            query, value = kwargs["q"], kwargs["v"]
            pointers = int(kwargs.get("num_k_exclude_rope", 0))
            if query.ndim != 3 or value.ndim != 3 or query.shape[0] != 1 or value.shape[0] != 1:
                raise RuntimeError("Expected batch-first SAM2 attention tensors")
            spatial = value.shape[1] - pointers
            labels = self.identity.labels
            if labels is None or labels.shape != (1, spatial):
                label_shape = None if labels is None else tuple(labels.shape)
                raise RuntimeError(
                    f"Region labels {label_shape} do not align with {spatial} spatial keys"
                )
            if spatial % (64 * 64) != 0:
                raise RuntimeError("Spatial keys and pointer tokens are misaligned")
            memory_features = module.k_proj(value[:, :spatial]).transpose(0, 1)
            query_features = module.q_proj(query)
            self.pending[layer] = (query_features, memory_features,
                                   labels, pointers)
        return before

    def _after(self, layer):
        def after(module, args, output):
            if layer not in self.pending:
                raise RuntimeError("Missing pre-projection matching features")
            query, memory, labels, pointers = self.pending.pop(layer)
            evidence, present = regional_matching_evidence(
                query, memory, labels, chunks=self.chunks
            )
            native_map = self._feature_map(output.transpose(1, 2))
            adapted = self.heads.layers[layer](native_map, evidence)
            self.calls.append({"layer": layer, "spatial_keys": int(memory.shape[0]),
                               "pointer_keys": pointers,
                               "present_regions": present.detach().cpu().tolist(),
                               "evidence_mean": evidence.detach().float().mean(dim=(0, 2, 3)).cpu().tolist(),
                               "evidence_min": evidence.detach().float().amin(dim=(0, 2, 3)).cpu().tolist(),
                               "evidence_max": evidence.detach().float().amax(dim=(0, 2, 3)).cpu().tolist(),
                               "evidence_std": evidence.detach().float().std(dim=(0, 2, 3), unbiased=False).cpu().tolist()})
            return adapted.flatten(2).transpose(1, 2)
        return after

    def __enter__(self):
        self.observer.__enter__()
        self.handles = []
        for index, layer in enumerate(self.model.memory_attention.layers):
            module = layer.cross_attn_image
            self.handles.extend([
                module.register_forward_pre_hook(self._before(index), with_kwargs=True),
                module.register_forward_hook(self._after(index)),
            ])
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.observer.__exit__(*exc)
