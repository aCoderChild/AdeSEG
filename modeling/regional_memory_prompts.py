"""Add soft region-conditioned summaries to frozen SAM2's native memory read."""

import inspect
import math

import torch
from torch import nn

from modeling.prediction_conditioned_affinity import memory_role_sequence
from modeling.simvos_region_prototypes import author_token_learner


class RegionalMemoryPrompts(nn.Module):
    def __init__(self, channels=64, tokens_per_region=16):
        super().__init__()
        learner = author_token_learner()
        self.learners = nn.ModuleList(
            learner(channels, tokens_per_region) for _ in range(3)
        )

    def forward(self, keys, values, memberships, spans, mode="regional"):
        if mode not in ("regional", "global"):
            raise ValueError("Expected regional or global conditioning")
        if keys.shape != values.shape or values.ndim != 3:
            raise ValueError("Expected matching batch-first memory keys and values")
        if memberships.shape != (*values.shape[:2], 3):
            raise ValueError("Stored region memberships do not align with memory")
        if not torch.isfinite(memberships).all() or (memberships < 0).any():
            raise ValueError("Invalid region memberships")
        if not torch.allclose(memberships.sum(-1), torch.ones_like(memberships[..., 0]), atol=1e-5):
            raise ValueError("Region memberships must sum to one")
        prompts, positions = [], []
        cursor = 0
        for span in spans:
            start, stop = span["start"], span["stop"]
            side = math.isqrt(stop - start)
            if start != cursor or side * side != stop - start:
                raise ValueError("Memory frames must be contiguous square grids")
            cursor = stop
            features = values[:, start:stop].float()
            pos = (keys[:, start:stop] - values[:, start:stop]).float()
            roles = memberships[:, start:stop].to(features)
            if mode == "global":
                roles = roles.mean(1, keepdim=True).expand_as(roles)
            for region, learner in enumerate(self.learners):
                weights = roles[..., region:region + 1]
                inputs = (features * weights).reshape(-1, side, side, features.shape[-1])
                # All-one author masks: no confidence-based suppression or selection.
                full = features.new_ones(features.shape[0], 1, side, side)
                summary, assignment = learner(inputs, full)
                prompts.append(summary)
                positions.append(torch.einsum("bpn,bnc->bpc", assignment, pos * weights))
        if cursor != values.shape[1] or not prompts:
            raise ValueError("Frame spans must cover every spatial memory token")
        value = torch.cat(prompts, 1).to(values.dtype)
        key = value + torch.cat(positions, 1).to(keys.dtype)
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise RuntimeError("Nonfinite memory prompts")
        return key, value


class NativeRegionalMemoryPrompts:
    def __init__(self, model, prompts, mode="regional", enabled=True):
        self.model, self.prompts = model, prompts
        self.mode, self.enabled = mode, enabled
        self.original = model._prepare_memory_conditioned_features
        self.signature = inspect.signature(self.original)
        self.memberships, self.spans = None, []
        self.frame = None
        self.calls, self.handles = [], []

    def _prepare(self, *args, **kwargs):
        data = self.signature.bind(*args, **kwargs)
        data.apply_defaults()
        data = data.arguments
        self.memberships, self.spans = None, []
        self.frame = int(data["frame_idx"])
        if self.enabled and not data["is_init_cond_frame"]:
            self.memberships, self.spans = memory_role_sequence(
                self.model, self.frame, data["output_dict"], data["num_frames"],
                data["track_in_reverse"], "regional",
            )
        return self.original(*args, **kwargs)

    def _before(self, layer):
        def hook(module, args, kwargs):
            if not self.enabled:
                return None
            if args or self.memberships is None:
                raise RuntimeError("Expected a native read with stored region memberships")
            key, value = kwargs["k"], kwargs["v"]
            pointers = int(kwargs.get("num_k_exclude_rope", 0))
            spatial = key.shape[1] - pointers
            prompt_key, prompt_value = self.prompts(
                key[:, :spatial], value[:, :spatial],
                self.memberships.to(key.device), self.spans, self.mode,
            )
            self.calls.append({"frame": self.frame, "layer": layer,
                               "spatial_keys": spatial, "pointer_keys": pointers,
                               "auxiliary_keys": prompt_key.shape[1],
                               "object_batch": key.shape[0], "memory_frames": self.spans})
            return args, {**kwargs, "k": torch.cat((key, prompt_key), 1),
                          "v": torch.cat((value, prompt_value), 1),
                          "num_k_exclude_rope": pointers + prompt_key.shape[1]}
        return hook

    def __enter__(self):
        if any(p.requires_grad for p in self.model.parameters()) or self.model.training:
            raise RuntimeError("All original SAM2 parameters must be frozen and in eval mode")
        layers = self.model.memory_attention.layers
        if len(layers) != 4:
            raise RuntimeError("Expected four native memory-attention layers")
        self.model._prepare_memory_conditioned_features = self._prepare
        for index, layer in enumerate(layers):
            self.handles.append(layer.cross_attn_image.register_forward_pre_hook(
                self._before(index), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.model._prepare_memory_conditioned_features = self.original
