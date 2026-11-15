"""Soft region-conditioned memory transforms built from Microsoft's LoRA layers."""

import inspect
from functools import lru_cache

import torch
from torch import nn

from modeling.prediction_conditioned_affinity import memory_role_sequence
from modeling.region_tokens import _verified_source

LORA_REVISION = "c4593f060e6a368d7bb5af5273b8e42810cdef90"


@lru_cache(maxsize=1)
def author_linear():
    path, source = _verified_source("LoRA", LORA_REVISION, "loralib/layers.py")
    namespace = {"__name__": "adeseg_author_loralib"}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["Linear"]


class RegionalLowRankLayer(nn.Module):
    def __init__(self, channels=64, rank=4):
        super().__init__()
        if channels <= 0 or not 0 < rank <= channels:
            raise ValueError("Expected positive channels and a valid low rank")
        linear = author_linear()
        self.regions = nn.ModuleList([
            linear(channels, channels, r=rank, lora_alpha=rank,
                   bias=False, merge_weights=False) for _ in range(3)
        ])
        with torch.no_grad():
            for region in self.regions:
                region.weight.zero_()

    def forward(self, memory, memberships, pointers):
        if memory.ndim != 3 or not 0 <= pointers < memory.shape[1]:
            raise ValueError("Expected batch-first memory and valid pointer count")
        spatial = memory.shape[1] - pointers
        if memberships.shape != (*memory.shape[:1], spatial, 3):
            raise ValueError("Memberships must align with spatial memory tokens")
        if not torch.isfinite(memberships).all() or (memberships < 0).any():
            raise ValueError("Memberships must be finite and nonnegative")
        if not torch.allclose(memberships.sum(-1), torch.ones_like(memberships[..., 0]), atol=1e-5):
            raise ValueError("Memberships must sum to one")
        values = memory[:, :spatial].to(self.regions[0].lora_A.dtype)
        residual = sum(region(values) * memberships[..., i:i + 1]
                       for i, region in enumerate(self.regions))
        adapted = memory[:, :spatial] + residual.to(memory.dtype)
        return torch.cat((adapted, memory[:, spatial:]), dim=1)


class RegionalLowRankHeads(nn.Module):
    def __init__(self, channels=64, rank=4):
        super().__init__()
        self.layers = nn.ModuleList(RegionalLowRankLayer(channels, rank) for _ in range(4))


class NativeRegionalLowRank:
    def __init__(self, model, heads, mode="regional", *, inference_state=None):
        if mode not in ("regional", "global"):
            raise ValueError("Expected regional or global control")
        self.model, self.heads, self.mode = model, heads, mode
        self.inference_state = inference_state
        self.object_id = None
        self.original = model._prepare_memory_conditioned_features
        self.signature = inspect.signature(self.original)
        self.memberships = None
        self.frame = None
        self.spans = []
        self.calls, self.handles = [], []

    def _prepare_read(self, *args, **kwargs):
        data = self.signature.bind(*args, **kwargs)
        data.apply_defaults()
        data = data.arguments
        self.memberships = None
        self.frame = int(data["frame_idx"])
        self.object_id = None
        if self.inference_state is not None:
            owners = [index for index, bank in self.inference_state["output_dict_per_obj"].items()
                      if bank is data["output_dict"]]
            if len(owners) != 1:
                raise RuntimeError("Memory bank does not belong to exactly one native object")
            self.object_id = self.inference_state["obj_idx_to_id"][owners[0]]
        if not data["is_init_cond_frame"]:
            self.memberships, self.spans = memory_role_sequence(
                self.model, self.frame, data["output_dict"], data["num_frames"],
                data["track_in_reverse"], self.mode,
            )
        return self.original(*args, **kwargs)

    def _before(self, index):
        def before(module, args, kwargs):
            if args or self.memberships is None:
                raise RuntimeError("Unexpected native memory-attention call")
            memory = kwargs["memory"]
            pointers = int(kwargs.get("num_k_exclude_rope", 0))
            roles = self.memberships.to(memory.device)
            adapted = self.heads.layers[index](memory, roles, pointers)
            if pointers and not torch.equal(adapted[:, -pointers:], memory[:, -pointers:]):
                raise RuntimeError("Pointer tokens were modified")
            self.calls.append({"frame": self.frame, "layer": index,
                               "object_id": self.object_id, "object_batch": roles.shape[0],
                               "spatial_keys": roles.shape[1], "pointer_keys": pointers,
                               "memory_frames": self.spans,
                               "pointer_tokens_unchanged": True})
            return args, {**kwargs, "memory": adapted}
        return before

    def __enter__(self):
        if len(self.model.memory_attention.layers) != 4:
            raise RuntimeError("Expected four native memory-attention layers")
        self.model._prepare_memory_conditioned_features = self._prepare_read
        for index, layer in enumerate(self.model.memory_attention.layers):
            self.handles.append(layer.register_forward_pre_hook(self._before(index), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.model._prepare_memory_conditioned_features = self.original
