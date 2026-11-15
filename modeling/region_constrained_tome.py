"""Region-constrained ToMe compression at SAM2's already-positioned memory read."""

from functools import lru_cache
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from modeling.prediction_conditioned_affinity import memory_role_sequence


ROOT = Path(__file__).resolve().parents[1]
TOME_REVISION = "af95e4b1befa172dadccd8c81e223b10090f9579"
TOME_SOURCE = "tome/merge.py"
WINDOW_SIZE = 4


@lru_cache(maxsize=1)
def author_tome():
    """Load the unmodified upstream ToMe matching and weighted-merge functions."""
    repo = ROOT / "external/ToMe"
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != TOME_REVISION:
        raise RuntimeError("ToMe checkout differs from the recorded author revision")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", TOME_SOURCE],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    source = repo / TOME_SOURCE
    spec = importlib.util.spec_from_file_location("_aadeseg_author_tome_merge", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the pinned ToMe source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tome_merge(values, metric, count):
    """Apply the author merge to one batch and return weighted tokens and masses."""
    tome = author_tome()
    merge, _ = tome.bipartite_soft_matching(metric, int(count))
    merged, size = tome.merge_wavg(merge, values)
    return merged, size


def _compress_group(keys, values, indices, count, matching_metric=None):
    """Merge one local group, sharing ToMe assignments across attention heads."""
    if count <= 0:
        mass = keys.new_ones((indices.numel(),), dtype=torch.float32)
        return keys.index_select(-2, indices), values.index_select(-2, indices), mass

    group_keys = keys.index_select(-2, indices)
    group_values = values.index_select(-2, indices)
    if not torch.isfinite(group_keys).all() or not torch.isfinite(group_values).all():
        raise ValueError("Memory keys and values must be finite before merging")
    metric = (group_keys.float().mean(dim=1) if matching_metric is None
              else matching_metric.index_select(-2, indices).float())
    if metric.shape[:2] != group_keys.shape[::2]:
        raise ValueError("Matching metric must align with batch and selected tokens")
    metric_norms = metric.norm(dim=-1)
    if not torch.isfinite(metric_norms).all() or (metric_norms <= 0).any():
        raise ValueError("ToMe matching metrics must have finite nonzero norms")
    reduced_keys, reduced_values, mass = [], [], None
    for head in range(keys.shape[1]):
        key_head = group_keys[:, head]
        value_head = group_values[:, head]
        merged_key, key_mass = _tome_merge(key_head, metric, count)
        merged_value, value_mass = _tome_merge(value_head, metric, count)
        if mass is None:
            mass = key_mass[0, :, 0].float()
        elif not torch.equal(mass, key_mass[0, :, 0].float()):
            raise RuntimeError("ToMe produced inconsistent key mass across heads")
        if not torch.equal(key_mass, value_mass):
            raise RuntimeError("ToMe key/value cluster sizes differ")
        reduced_keys.append(merged_key)
        reduced_values.append(merged_value)
    return (
        torch.stack(reduced_keys, dim=1),
        torch.stack(reduced_values, dim=1),
        mass,
    )


def compress_memory_kv(keys, values, memberships, spans, *, mode, pointers,
                       window_size=WINDOW_SIZE, merge_plan=None, merge_divisor=2):
    """Compress spatial K/V after native projections and RoPE, leaving pointers intact.

    ``memberships`` are the three continuous roles computed from each stored
    predicted mask. The ToMe matching step assigns each cell to its largest role
    only to forbid cross-role merges; this is positional bookkeeping, not a claim
    that a feature vector is semantically pure.
    """
    if mode not in ("regional", "global"):
        raise ValueError("Expected regional or global ToMe control")
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("Expected matching [batch, heads, keys, channels] tensors")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("Memory keys and values must be finite")
    batch, _, key_count, _ = keys.shape
    if batch != 1:
        raise ValueError("The first engineering gate supports one object batch")
    if pointers < 0 or pointers >= key_count:
        raise ValueError("Invalid native pointer count")
    spatial = key_count - pointers
    if memberships.shape != (batch, spatial, 3):
        raise ValueError("Predicted region roles do not match the spatial K/V keys")
    if not torch.isfinite(memberships).all() or (memberships < 0).any():
        raise ValueError("Region memberships must be finite and nonnegative")
    if not torch.allclose(memberships.sum(-1), torch.ones_like(memberships[..., 0]), atol=1e-5):
        raise ValueError("Region memberships must sum to one")
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if type(merge_divisor) is not int or merge_divisor < 2:
        raise ValueError("Merge divisor must be an integer of at least two")

    role_ids = memberships.argmax(dim=-1)[0]
    compressed_keys, compressed_values, masses = [], [], []
    frame_audit = []
    cursor = 0
    for span in spans:
        start, stop = int(span["start"]), int(span["stop"])
        side = int((stop - start) ** 0.5)
        if start != cursor or side * side != stop - start:
            raise ValueError("Memory spans must be contiguous square grids")
        if side % window_size:
            raise ValueError("Window size must evenly tile each memory grid")
        cursor = stop
        before = stop - start
        frame_merges = 0
        frame_windows = []
        plan_for_frame = None
        if merge_plan is not None:
            plan_for_frame = merge_plan.get(span["frame"])
            if plan_for_frame is None:
                raise ValueError(f"Missing matched merge plan for memory frame {span['frame']}")
        window_index = 0
        for top in range(0, side, window_size):
            for left in range(0, side, window_size):
                local = torch.arange(start, stop, device=keys.device).reshape(side, side)[
                    top:top + window_size, left:left + window_size
                ].reshape(-1)
                local_roles = role_ids.index_select(0, local)
                groups = [local[local_roles == region] for region in range(3)]
                regional_counts = [int(group.numel()) // merge_divisor for group in groups]
                target_merges = sum(regional_counts)
                if mode == "global" and plan_for_frame is not None:
                    if window_index not in plan_for_frame:
                        raise ValueError(
                            f"Missing window {window_index} in matched plan for frame {span['frame']}"
                        )
                    target_merges = int(plan_for_frame[window_index])
                    if target_merges < 0 or target_merges > local.numel() // merge_divisor:
                        raise ValueError("Matched merge count exceeds the local ToMe limit")
                frame_merges += target_merges
                frame_windows.append({
                    "window": window_index,
                    "original_spatial_keys": int(local.numel()),
                    "merges": target_merges,
                    "retained_spatial_keys": int(local.numel()) - target_merges,
                })
                window_index += 1

                if mode == "regional":
                    for group, count in zip(groups, regional_counts, strict=True):
                        k_out, v_out, size = _compress_group(keys, values, group, count)
                        compressed_keys.append(k_out)
                        compressed_values.append(v_out)
                        masses.append(size)
                else:
                    k_out, v_out, size = _compress_group(keys, values, local, target_merges)
                    compressed_keys.append(k_out)
                    compressed_values.append(v_out)
                    masses.append(size)

        frame_audit.append({
            "frame": span["frame"],
            "t_pos": int(span["t_pos"]),
            "original_spatial_keys": before,
            "merged_keys": frame_merges,
            "retained_spatial_keys": before - frame_merges,
            "region_token_counts": [int((role_ids[start:stop] == r).sum()) for r in range(3)],
            "windows": frame_windows,
        })
        if plan_for_frame is not None and len(plan_for_frame) != window_index:
            raise ValueError(f"Unexpected window count in matched plan for frame {span['frame']}")
    if cursor != spatial or not compressed_keys:
        raise ValueError("Memory spans do not cover all spatial keys")

    spatial_keys = torch.cat(compressed_keys, dim=-2)
    spatial_values = torch.cat(compressed_values, dim=-2)
    spatial_mass = torch.cat(masses).to(device=keys.device, dtype=torch.float32)
    pointer_keys = keys[..., spatial:, :]
    pointer_values = values[..., spatial:, :]
    out_keys = torch.cat((spatial_keys, pointer_keys), dim=-2)
    out_values = torch.cat((spatial_values, pointer_values), dim=-2)
    all_mass = torch.cat((spatial_mass, spatial_mass.new_ones(pointers)))
    if int(all_mass.numel()) != out_keys.shape[-2] or not torch.isfinite(all_mass).all():
        raise RuntimeError("Compressed key mass is invalid")
    if pointers and not torch.equal(out_keys[..., -pointers:, :], keys[..., -pointers:, :]):
        raise RuntimeError("ToMe modified native pointer keys")
    if pointers and not torch.equal(out_values[..., -pointers:, :], values[..., -pointers:, :]):
        raise RuntimeError("ToMe modified native pointer values")
    expected_spatial_mass = torch.tensor(float(spatial), device=keys.device)
    if not torch.allclose(spatial_mass.sum(), expected_spatial_mass, atol=1e-5, rtol=0):
        raise RuntimeError("Merged tokens do not conserve the original key count")

    audit = {
        "mode": mode,
        "spatial_keys_before": spatial,
        "spatial_keys_after": int(spatial_keys.shape[-2]),
        "pointer_keys": pointers,
        "spatial_key_mass_sum": float(spatial_mass.sum()),
        "key_mass_sum": float(all_mass.sum()),
        "frames": frame_audit,
    }
    attention_bias = all_mass.log().view(1, 1, 1, -1)
    if spatial_keys.shape[-2] == spatial:
        # A zero-removal gate must preserve key order as well as values.
        return keys, values, attention_bias, audit
    return out_keys, out_values, attention_bias, audit


class NativeRegionConstrainedToMe:
    """Apply ToMe at the native cross-attention call without changing SAM2 weights."""

    def __init__(self, model, mode="regional", window_size=WINDOW_SIZE,
                 merge_plan_by_target=None, object_id_by_output_dict=None,
                 passthrough=False, merge_divisor=2):
        if mode not in ("regional", "global"):
            raise ValueError("Expected regional or global ToMe control")
        self.model, self.mode, self.window_size = model, mode, int(window_size)
        if type(merge_divisor) is not int or merge_divisor < 2:
            raise ValueError("Merge divisor must be an integer of at least two")
        self.merge_divisor = merge_divisor
        self.requires_roles = getattr(self, "requires_roles", True)
        self.merge_plan_by_target = merge_plan_by_target
        self.object_id_by_output_dict = object_id_by_output_dict
        self.current_object_id = None
        self.passthrough = bool(passthrough)
        self.original_prepare = model._prepare_memory_conditioned_features
        self.prepare_signature = inspect.signature(self.original_prepare)
        self.memberships, self.spans = None, []
        self.frame = None
        self.active = None
        self.inside = False
        self.calls, self.handles = [], []

    def _compress_memory(self, keys, values, memberships, spans, pointers, merge_plan):
        return compress_memory_kv(
            keys, values, memberships, spans,
            mode=self.mode, pointers=pointers, window_size=self.window_size,
            merge_plan=merge_plan, merge_divisor=self.merge_divisor,
        )

    def _prepare(self, *args, **kwargs):
        bound = self.prepare_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        data = bound.arguments
        previous_object_id = self.current_object_id
        if self.object_id_by_output_dict is not None:
            try:
                self.current_object_id = self.object_id_by_output_dict[id(data["output_dict"])]
            except KeyError as exc:
                raise RuntimeError("SAM2 used an unregistered per-object memory bank") from exc
        self.memberships, self.spans = None, []
        self.frame = int(data["frame_idx"])
        try:
            if not data["is_init_cond_frame"]:
                if self.requires_roles:
                    self.memberships, self.spans = memory_role_sequence(
                        self.model, self.frame, data["output_dict"], data["num_frames"],
                        data["track_in_reverse"], "regional",
                    )
                else:
                    from modeling.native_memory_layout import selected_memory_frames
                    self.spans = selected_memory_frames(
                        self.model, self.frame, data["output_dict"], data["num_frames"],
                        data["track_in_reverse"],
                    )
            return self.original_prepare(*args, **kwargs)
        finally:
            self.current_object_id = previous_object_id

    def _before_attention(self, layer):
        def before(module, args, kwargs):
            if self.active is not None:
                raise RuntimeError("Nested memory cross-attention call")
            self.active = (layer, int(kwargs.get("num_k_exclude_rope", 0)))
        return before

    def _after_attention(self, module, args, kwargs, output):
        self.active = None

    def _sdpa(self, query, key, value, *args, **kwargs):
        if self.inside or self.active is None:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        if args or kwargs.get("attn_mask") is not None or kwargs.get("dropout_p", 0.0) != 0.0:
            raise RuntimeError("Unsupported SAM2 memory SDPA options")
        if kwargs.get("is_causal", False):
            raise RuntimeError("Causal SDPA is not expected in SAM2 memory attention")
        layer, pointers = self.active
        if self.passthrough:
            output = self.original_sdpa(query, key, value, *args, **kwargs)
            self.calls.append({
                "frame": self.frame,
                "object_id": self.current_object_id,
                "layer": layer,
                "mode": "passthrough",
                "spatial_keys_before": key.shape[-2] - pointers,
                "spatial_keys_after": key.shape[-2] - pointers,
                "pointer_keys": pointers,
                "frames": [
                    {"frame": int(span["frame"]), "t_pos": int(span["t_pos"])}
                    for span in self.spans
                ],
            })
            return output
        if self.requires_roles and self.memberships is None:
            raise RuntimeError("Memory-role labels were not prepared for this read")
        plan_key = self.frame
        if self.object_id_by_output_dict is not None:
            plan_key = (self.frame, self.current_object_id)
        merge_plan = None
        if self.merge_plan_by_target is not None:
            if plan_key not in self.merge_plan_by_target:
                raise RuntimeError(f"Missing matched token budget for target {plan_key}")
            merge_plan = self.merge_plan_by_target[plan_key]
        key_reduced, value_reduced, mass_bias, audit = self._compress_memory(
            key, value,
            self.memberships.to(key.device) if self.memberships is not None else None,
            self.spans, pointers, merge_plan,
        )
        self.inside = True
        try:
            if audit['spatial_keys_after'] == audit['spatial_keys_before']:
                output = self.original_sdpa(query, key, value, *args, **kwargs)
            else:
                output = self.original_sdpa(
                    query, key_reduced, value_reduced,
                    attn_mask=mass_bias.to(dtype=query.dtype),
                    dropout_p=0.0,
                    is_causal=False,
                    scale=kwargs.get("scale"),
                    enable_gqa=kwargs.get("enable_gqa", False),
                )
        finally:
            self.inside = False
        self.calls.append({
            "frame": self.frame,
            "object_id": self.current_object_id,
            "layer": layer,
            **audit,
        })
        return output

    def __enter__(self):
        if any(parameter.requires_grad for parameter in self.model.parameters()) or self.model.training:
            raise RuntimeError("SAM2 must be frozen and in eval mode")
        if len(self.model.memory_attention.layers) != 4:
            raise RuntimeError("Expected all four native SAM2 memory-attention layers")
        self.model._prepare_memory_conditioned_features = self._prepare
        for index, layer in enumerate(self.model.memory_attention.layers):
            attention = layer.cross_attn_image
            self.handles.append(attention.register_forward_pre_hook(
                self._before_attention(index), with_kwargs=True
            ))
            self.handles.append(attention.register_forward_hook(
                self._after_attention, with_kwargs=True, always_call=True
            ))
        self.original_sdpa = F.scaled_dot_product_attention
        from unittest.mock import patch
        self.override = patch.object(F, "scaled_dot_product_attention", self._sdpa)
        self.override.start()
        return self

    def __exit__(self, *exc):
        self.override.stop()
        for handle in self.handles:
            handle.remove()
        self.model._prepare_memory_conditioned_features = self.original_prepare
        self.active = None
