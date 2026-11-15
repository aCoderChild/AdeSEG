"""Add a learned SimVOS prototype read beside native SAM2 memory attention."""

import inspect
from unittest.mock import patch

import torch
import torch.nn.functional as F

from modeling.region_definitions import mixed_cell_regions
from modeling.region_bias import feature_key


class NativeSimVOSPrototypeRead:
    """Pool exact SAM2 memory blocks with author TokenLearner assignments."""
    def __init__(self, model, heads, mode, sequence):
        if mode not in ("regional", "global"):
            raise ValueError("Expected regional or global prototype control")
        self.model, self.heads, self.mode, self.sequence = model, heads, mode, sequence
        self.prepare = model._prepare_memory_conditioned_features
        self.signature = inspect.signature(self.prepare)
        self.bank = {}
        self.current = None
        self.frame = None
        self.active = None
        self.calls = []

    def _prepare_read(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        data = bound.arguments
        self.frame = int(data["frame_idx"])
        self.bank = {}
        if not data["is_init_cond_frame"]:
            for outputs in data["output_dict"].values():
                for index, stored in outputs.items():
                    if index >= self.frame or stored.get("maskmem_features") is None:
                        continue
                    key = feature_key(stored["maskmem_features"].flatten(2).transpose(1, 2))
                    labels = mixed_cell_regions(stored["pred_masks"][0, 0])["labels"]
                    if key in self.bank:
                        raise RuntimeError("Ambiguous stored memory identity")
                    self.bank[key] = (int(index), torch.as_tensor(labels).long())
        return self.prepare(*args, **kwargs)

    def _capture_memory(self, module, args, kwargs):
        if args:
            raise RuntimeError("Unexpected memory-attention signature")
        memory = kwargs["memory"]
        memory_pos = kwargs["memory_pos"]
        pointers = int(kwargs.get("num_obj_ptr_tokens", 0))
        if memory.ndim != 3 or memory.shape[1] != 1 or memory.shape[-1] != 64:
            raise RuntimeError("Expected sequence-first, single-object 64-channel memory")
        if memory_pos.shape != memory.shape:
            raise RuntimeError("Memory and position shapes differ")

        items = []
        spatial_keys = memory.shape[0] - pointers
        if spatial_keys <= 0 or spatial_keys % 4096:
            raise RuntimeError("Unexpected SAM2 spatial-memory token count")
        memory_frames = []
        for start in range(0, spatial_keys, 4096):
            end = start + 4096
            frame_values = memory[start:end].transpose(0, 1)
            identity = feature_key(frame_values)
            if identity not in self.bank:
                raise RuntimeError("Memory block does not match a stored prediction")
            index, labels = self.bank[identity]
            if index >= self.frame:
                raise RuntimeError("Current or future memory frame was selected")
            memory_frames.append(index)
            labels = labels.to(device=memory.device)
            features = frame_values.reshape(1, 64, 64, 64).float()
            pooled = self.heads.assign(features, labels, self.mode)
            for region, prototypes, assignment in pooled:
                items.append({
                    "frame": index,
                    "region": region,
                    "features": prototypes,
                    "assignment": assignment,
                    "start": start,
                    "end": end,
                })
        if spatial_keys + pointers != len(memory):
            raise RuntimeError("Spatial/pointer counts do not match the native bank")
        self.current = {
            "items": items,
            "spatial_keys": spatial_keys,
            "pointer_keys": pointers,
            "memory_frames": memory_frames,
        }

    def _before_attention(self, layer):
        def before(module, args, kwargs):
            if self.active is not None:
                raise RuntimeError("Nested SAM2 cross-attention call")
            self.active = layer
        return before

    def _after_attention(self, module, args, kwargs, output):
        self.active = None

    def _attention(self, q, k, v, *args, **kwargs):
        native = self.sdpa(q, k, v, *args, **kwargs)
        if self.active is None or self.current is None or not self.current["items"]:
            return native
        if args or kwargs.get("attn_mask") is not None or kwargs.get("dropout_p", 0) != 0:
            raise RuntimeError("Prototype read expects native unmasked zero-dropout SDPA")

        layer = self.active
        attention = self.model.memory_attention.layers[layer].cross_attn_image
        proto_keys, proto_values = [], []
        for item in self.current["items"]:
            assignment = item["assignment"].to(device=k.device, dtype=torch.float32)
            start, end = item["start"], item["end"]
            region_key = torch.einsum(
                "bpn,bhnd->bhpd", assignment, k[:, :, start:end].float()
            )
            projected_value = attention.v_proj(
                item["features"].to(dtype=attention.v_proj.weight.dtype)
            )
            region_value = attention._separate_heads(
                projected_value, attention.num_heads
            )
            proto_keys.append(region_key.to(dtype=q.dtype))
            proto_values.append(region_value.to(dtype=q.dtype))
        proto_key = torch.cat(proto_keys, dim=-2)
        proto_value = torch.cat(proto_values, dim=-2)
        prototype_read = self.sdpa(q, proto_key, proto_value, dropout_p=0.0)
        gain = self.heads.gains[layer].to(dtype=native.dtype)
        if not torch.isfinite(prototype_read).all():
            raise RuntimeError("Prototype attention returned nonfinite values")
        result = native + gain * prototype_read
        self.calls.append({
            "frame": self.frame,
            "layer": layer,
            "memory_frames": self.current["memory_frames"],
            "spatial_keys": self.current["spatial_keys"],
            "pointer_keys": self.current["pointer_keys"],
            "prototype_keys": int(proto_key.shape[-2]),
            "prototype_regions": sorted({item["region"] for item in self.current["items"]}),
            "gain": float(gain.detach().cpu()),
            "prototype_read_rms": float(prototype_read.detach().float().square().mean().sqrt().cpu()),
        })
        return result

    def __enter__(self):
        memory_attention = self.model.memory_attention
        if getattr(memory_attention, "utility_controller", None) is not None or getattr(
            memory_attention, "capture_attention", False
        ):
            raise RuntimeError("Disable other memory interventions and attention capture")
        self.prepare_patch = patch.object(
            self.model, "_prepare_memory_conditioned_features", self._prepare_read
        )
        self.prepare_patch.start()
        self.handles = [memory_attention.register_forward_pre_hook(self._capture_memory, with_kwargs=True)]
        for index, layer in enumerate(memory_attention.layers):
            attention = layer.cross_attn_image
            self.handles.append(attention.register_forward_pre_hook(self._before_attention(index), with_kwargs=True))
            self.handles.append(attention.register_forward_hook(self._after_attention, with_kwargs=True, always_call=True))
        self.sdpa = F.scaled_dot_product_attention
        self.override = patch.object(F, "scaled_dot_product_attention", self._attention)
        self.override.start()
        return self

    def __exit__(self, *exc):
        self.override.stop()
        for handle in self.handles:
            handle.remove()
        self.prepare_patch.stop()
        self.current = None
        self.active = None
