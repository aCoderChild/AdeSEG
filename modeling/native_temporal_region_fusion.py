"""Read all SAM2 memory tokens, then apply a zero-initialized regional residual."""

from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from modeling.cfbi_region_gate import NativeCFBIGate
from modeling.region_readout import regional_readouts
from modeling.temporal_region_fusion import TemporalRegionFusion


class _RegionLabels(nn.Module):
    def region_context(self, vectors, labels, mode):
        if mode != "regional":
            raise ValueError("Only predicted-mask regional labels are supported")
        self.labels = labels[None]
        return vectors.new_zeros((1, 192))

    def forward(self, attention, context):
        return attention


class TemporalRegionHeads(nn.Module):
    def __init__(self, channels=256, layers=4):
        super().__init__()
        self.layers = nn.ModuleList(TemporalRegionFusion(channels) for _ in range(layers))


class NativeTemporalRegionFusion:
    def __init__(self, model, heads, sequence):
        self.model, self.heads, self.sequence = model, heads, sequence
        self.labels = _RegionLabels()
        self.observer = NativeCFBIGate(model, self.labels, "regional", sequence, 0)
        self.active = None
        self.in_adapter = False
        self.calls = []

    @staticmethod
    def _feature_map(tensor, height=64, width=64):
        batch, heads, tokens, dim = tensor.shape
        if tokens != height * width:
            raise RuntimeError(f"Expected a {height}x{width} memory query, got {tokens} tokens")
        return tensor.permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim).transpose(1, 2).reshape(
            batch, heads * dim, height, width
        )

    @staticmethod
    def _attention_map(feature, heads, head_dim):
        batch, channels, height, width = feature.shape
        if channels != heads * head_dim:
            raise RuntimeError("Regional residual channels do not match SAM2 attention")
        return feature.flatten(2).transpose(1, 2).reshape(batch, height * width, heads, head_dim).transpose(1, 2)

    def _before(self, layer):
        def before(module, args, kwargs):
            if self.active is not None:
                raise RuntimeError("Nested memory attention")
            self.active = (layer, int(kwargs.get("num_k_exclude_rope", 0)))
        return before

    def _after(self, module, args, kwargs, output):
        self.active = None

    def _attention(self, query, key, value, *args, **kwargs):
        if self.in_adapter:
            return self.sdpa(query, key, value, *args, **kwargs)
        native = self.sdpa(query, key, value, *args, **kwargs)
        if self.active is None:
            return native
        if args or kwargs.get("attn_mask") is not None or kwargs.get("dropout_p", 0) != 0:
            raise RuntimeError("Unsupported SAM2 memory-attention parameters")
        layer, pointers = self.active
        labels = self.labels.labels
        spatial = key.shape[-2] - pointers
        frame_size = 64 * 64
        if spatial <= 0 or spatial % frame_size or labels.shape != (1, spatial):
            raise RuntimeError("Memory keys and stored-mask region labels are misaligned")
        frame_records = self.observer.trace[-1]["memory_frames"]
        if len(frame_records) * frame_size != spatial:
            raise RuntimeError("Stored frame manifest does not explain the spatial memory keys")
        recent_index = frame_records.index(max(frame_records))
        start, stop = recent_index * frame_size, (recent_index + 1) * frame_size
        recent_label = labels[:, start:stop].reshape(1, 1, 64, 64)

        self.in_adapter = True
        try:
            reads, valid = regional_readouts(query, key, value, labels, pointers, sdpa=self.sdpa)
            native_map = self._feature_map(native)
            regional_map = torch.stack([self._feature_map(reads[:, region]) for region in range(3)], dim=1)
            recent_values = value[..., start:stop, :]
            recent_map = self._feature_map(recent_values)
            query_map = self._feature_map(query)
            present = valid.to(regional_map.dtype).view(1, 3, 1, 1, 1)
            region_mean = (regional_map * present).sum(dim=1) / present.sum(dim=1).clamp_min(1)
            difference = region_mean - recent_map
            output_map, uncertainty = self.heads.layers[layer](
                native_map,
                regional_map,
                recent_map,
                query_map,
                (recent_label != 0).to(native_map.dtype),
                difference,
            )
            output = self._attention_map(output_map, query.shape[1], query.shape[-1])
        finally:
            self.in_adapter = False
        self.calls.append({
            "layer": layer,
            "spatial_keys": spatial,
            "pointer_keys": pointers,
            "memory_frames": frame_records,
            "valid_regions": valid.detach().cpu().tolist(),
            "uncertainty_mean": float(uncertainty.detach().float().mean()),
        })
        return output

    def __enter__(self):
        self.observer.__enter__()
        self.handles = []
        for index, layer in enumerate(self.model.memory_attention.layers):
            attention = layer.cross_attn_image
            self.handles.append(attention.register_forward_pre_hook(self._before(index), with_kwargs=True))
            self.handles.append(attention.register_forward_hook(self._after, with_kwargs=True, always_call=True))
        self.sdpa = F.scaled_dot_product_attention
        self.override = patch.object(F, "scaled_dot_product_attention", self._attention)
        self.override.start()
        return self

    def __exit__(self, *exc):
        self.override.stop()
        for handle in self.handles:
            handle.remove()
        self.observer.__exit__(*exc)
