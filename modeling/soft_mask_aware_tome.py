"""ToMe memory merging with continuous mask-occupancy compatibility.

This is a separate exploratory variant. It reuses the pinned upstream ToMe
matching and weighted-merge functions through ``region_constrained_tome`` and
does not alter SAM2 parameters or the active region-constrained ToMe run.
"""

import hashlib
import math

import torch
import torch.nn.functional as F

from modeling.prediction_conditioned_affinity import memory_role_sequence
from modeling.region_constrained_tome import (
    NativeRegionConstrainedToMe,
    _tome_merge,
)


WINDOW_SIZE = 4
MERGES_PER_WINDOW = 8
ROLE_WEIGHT = 0.5


def occupancy_from_mask_logits(mask_logits, grid_shape, *, device=None):
    """Area-average native high-resolution mask probabilities onto memory grid.

    ``mask_logits`` must be the exact high-resolution prediction stored for the
    frame and used to form SAM2's memory-encoder mask input. No binary threshold
    or three-way region assignment is applied.
    """
    logits = torch.as_tensor(mask_logits).detach().float()
    while logits.ndim > 2 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite single-object stored-mask logits")
    if len(grid_shape) != 2:
        raise ValueError("Memory grid shape must contain height and width")
    grid_h, grid_w = (int(value) for value in grid_shape)
    mask_h, mask_w = logits.shape
    if grid_h <= 0 or grid_w <= 0 or mask_h % grid_h or mask_w % grid_w:
        raise ValueError("Mask dimensions must tile the native memory grid exactly")
    pool_kernel = (mask_h // grid_h, mask_w // grid_w)
    probability = F.avg_pool2d(
        logits.sigmoid()[None, None], kernel_size=pool_kernel, stride=pool_kernel
    )[0, 0]
    if probability.shape != (grid_h, grid_w):
        raise RuntimeError("Pooled occupancy differs from the requested memory grid")
    if device is not None:
        probability = probability.to(device=device)
    return probability.reshape(-1)


def soft_mask_similarity_metric(keys, occupancy, role_weight=ROLE_WEIGHT):
    """Build ToMe's matching metric from appearance and soft occupancy.

    Appending a normalized two-component occupancy embedding makes the
    resulting cosine similarity an explicit convex blend of appearance
    similarity and soft object/background-occupancy similarity. Intermediate
    occupancies remain compatible with both sides of a predicted contour.
    """
    if keys.ndim != 4:
        raise ValueError("Expected memory keys [batch, heads, tokens, channels]")
    if not 0.0 <= role_weight <= 1.0:
        raise ValueError("role_weight must be in [0, 1]")
    batch, _, tokens, _ = keys.shape
    if occupancy.shape != (batch, tokens):
        raise ValueError("Occupancy must align with the memory spatial keys")
    if not torch.isfinite(keys).all() or not torch.isfinite(occupancy).all():
        raise ValueError("Matching inputs must be finite")
    if (occupancy < 0).any() or (occupancy > 1).any():
        raise ValueError("Occupancy must be in [0, 1]")

    appearance = keys.float().mean(dim=1)
    appearance_norm = appearance.norm(dim=-1, keepdim=True)
    if (appearance_norm <= 0).any():
        raise ValueError("Appearance matching vectors must have nonzero norm")
    appearance = appearance / appearance_norm

    p = occupancy.float()
    role = torch.stack((1.0 - p, p), dim=-1)
    role = F.normalize(role, p=2, dim=-1)
    return torch.cat(
        (
            appearance * math.sqrt(1.0 - role_weight),
            role * math.sqrt(role_weight),
        ),
        dim=-1,
    )


def _merge_window(keys, values, indices, occupancy, *, mode, merges, role_weight):
    if merges < 0 or merges > indices.numel() // 2:
        raise ValueError("Per-window merge count is outside the ToMe limit")
    group_keys = keys.index_select(-2, indices)
    group_values = values.index_select(-2, indices)
    if merges == 0:
        mass = keys.new_ones((indices.numel(),), dtype=torch.float32)
        return group_keys, group_values, mass

    p = occupancy.index_select(1, indices)
    if mode == "soft_mask_aware":
        metric = soft_mask_similarity_metric(group_keys, p, role_weight)
    elif mode == "generic":
        metric = group_keys.float().mean(dim=1)
    else:
        raise ValueError("mode must be 'soft_mask_aware' or 'generic'")

    norm = metric.norm(dim=-1)
    if not torch.isfinite(metric).all() or (norm <= 0).any():
        raise ValueError("ToMe matching metric must have finite nonzero norms")

    merged_keys, merged_values, mass = [], [], None
    for head in range(keys.shape[1]):
        key_head, value_head = group_keys[:, head], group_values[:, head]
        key_out, key_mass = _tome_merge(key_head, metric, merges)
        value_out, value_mass = _tome_merge(value_head, metric, merges)
        if not torch.equal(key_mass, value_mass):
            raise RuntimeError("Key/value ToMe assignments have different cluster sizes")
        this_mass = key_mass[0, :, 0].float()
        if mass is None:
            mass = this_mass
        elif not torch.equal(mass, this_mass):
            raise RuntimeError("ToMe cluster sizes differ across attention heads")
        merged_keys.append(key_out)
        merged_values.append(value_out)
    return torch.stack(merged_keys, dim=1), torch.stack(merged_values, dim=1), mass


def compress_memory_kv(keys, values, occupancy, spans, *, mode, pointers,
                       window_size=WINDOW_SIZE, merges_per_window=MERGES_PER_WINDOW,
                       role_weight=ROLE_WEIGHT):
    """Merge equal fixed budgets in local memory windows and preserve pointers."""
    if mode not in ("soft_mask_aware", "generic"):
        raise ValueError("Unknown soft-mask ToMe mode")
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("Expected matching K/V tensors [batch, heads, keys, channels]")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("Memory keys and values must be finite")
    batch, _, key_count, _ = keys.shape
    if batch != 1:
        raise ValueError("The current video predictor gate supports one object batch")
    if pointers < 0 or pointers >= key_count:
        raise ValueError("Invalid pointer key count")
    spatial = key_count - pointers
    if occupancy.shape != (batch, spatial):
        raise ValueError("Occupancy does not align with spatial K/V keys")
    if not torch.isfinite(occupancy).all() or (occupancy < 0).any() or (occupancy > 1).any():
        raise ValueError("Occupancy values must be finite and lie in [0, 1]")
    if window_size <= 0 or merges_per_window < 0:
        raise ValueError("Window and merge counts must be nonnegative")

    out_keys, out_values, masses = [], [], []
    frame_audits = []
    cursor = 0
    for span in spans:
        start, stop = int(span["start"]), int(span["stop"])
        side = int((stop - start) ** 0.5)
        if start != cursor or side * side != stop - start or side % window_size:
            raise ValueError("Memory spans must be contiguous square grids tiled by windows")
        cursor = stop
        windows = []
        grid = torch.arange(start, stop, device=keys.device).reshape(side, side)
        for window_id, (top, left) in enumerate(
            ( (top, left) for top in range(0, side, window_size)
              for left in range(0, side, window_size) )
        ):
            indices = grid[top:top + window_size, left:left + window_size].reshape(-1)
            count = min(int(merges_per_window), indices.numel() // 2)
            k_out, v_out, mass = _merge_window(
                keys, values, indices, occupancy,
                mode=mode, merges=count, role_weight=role_weight,
            )
            out_keys.append(k_out)
            out_values.append(v_out)
            masses.append(mass)
            windows.append({
                "window": window_id,
                "original_spatial_keys": int(indices.numel()),
                "merges": count,
                "retained_spatial_keys": int(indices.numel()) - count,
            })
        frame_audits.append({"frame": int(span["frame"]), "windows": windows})

    if cursor != spatial or not out_keys:
        raise ValueError("Memory spans do not cover every spatial key")
    spatial_keys = torch.cat(out_keys, dim=-2)
    spatial_values = torch.cat(out_values, dim=-2)
    spatial_mass = torch.cat(masses).to(device=keys.device, dtype=torch.float32)
    pointer_keys = keys[..., spatial:, :]
    pointer_values = values[..., spatial:, :]
    merged_keys = torch.cat((spatial_keys, pointer_keys), dim=-2)
    merged_values = torch.cat((spatial_values, pointer_values), dim=-2)
    all_mass = torch.cat((spatial_mass, spatial_mass.new_ones(pointers)))

    if pointers and (
        not torch.equal(merged_keys[..., -pointers:, :], keys[..., -pointers:, :])
        or not torch.equal(merged_values[..., -pointers:, :], values[..., -pointers:, :])
    ):
        raise RuntimeError("Object pointer K/V must remain bit-identical")
    if not torch.isfinite(all_mass).all() or (all_mass <= 0).any():
        raise RuntimeError("Merged token masses must be finite and positive")
    if int(all_mass.numel()) != merged_keys.shape[-2]:
        raise RuntimeError("Token mass count differs from compressed K/V length")
    if not torch.allclose(spatial_mass.sum(), torch.tensor(float(spatial), device=keys.device), atol=1e-5, rtol=0):
        raise RuntimeError("Merge operation did not conserve spatial key mass")

    audit = {
        "mode": mode,
        "spatial_keys_before": spatial,
        "spatial_keys_after": int(spatial_keys.shape[-2]),
        "pointer_keys": pointers,
        "spatial_key_mass_sum": float(spatial_mass.sum()),
        "key_mass_sum": float(all_mass.sum()),
        "frames": frame_audits,
    }
    attention_bias = all_mass.log().view(1, 1, 1, -1)
    return merged_keys, merged_values, attention_bias, audit


class NativeSoftMaskAwareToMe(NativeRegionConstrainedToMe):
    """Apply mask-aware or appearance-only ToMe at native SAM2 memory SDPA."""

    def __init__(self, model, mode="soft_mask_aware", *, role_weight=ROLE_WEIGHT):
        if mode not in ("soft_mask_aware", "generic"):
            raise ValueError("Unknown soft-mask ToMe mode")
        parent_mode = "regional" if mode == "soft_mask_aware" else "global"
        super().__init__(model, mode=parent_mode)
        self.method_mode = mode
        self.role_weight = float(role_weight)
        self.occupancy = None
        self.mask_sources = []
        self.mask_cache = {}

    def _prepare(self, *args, **kwargs):
        bound = self.prepare_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        data = bound.arguments
        self.occupancy, self.spans, self.mask_sources = None, [], []
        self.frame = int(data["frame_idx"])
        if not data["is_init_cond_frame"]:
            _, self.spans = memory_role_sequence(
                self.model,
                self.frame,
                data["output_dict"],
                data["num_frames"],
                data["track_in_reverse"],
                "regional",
            )
            outputs = {}
            for name in ("cond_frame_outputs", "non_cond_frame_outputs"):
                outputs.update({int(index): output for index, output in data["output_dict"][name].items()})
            maps = []
            for span in self.spans:
                output = outputs.get(int(span["frame"]))
                if output is None:
                    raise RuntimeError(f"Missing stored prediction for memory frame {span['frame']}")
                mask_field = "pred_masks_high_res"
                masks = output.get(mask_field)
                if not isinstance(masks, torch.Tensor) or masks.ndim != 4 or masks.shape[:2] != (1, 1):
                    raise ValueError("Expected one stored high-resolution object mask per frame")
                memory_features = output.get("maskmem_features")
                if not isinstance(memory_features, torch.Tensor) or memory_features.ndim != 4:
                    raise ValueError("Stored frame is missing spatial memory features")
                memory_grid_shape = tuple(int(value) for value in memory_features.shape[-2:])
                source_frame = int(span["frame"])
                cached = self.mask_cache.get(source_frame)
                if cached is None:
                    native_mask = masks[0, 0].detach().float().cpu().contiguous()
                    mask_sha256 = hashlib.sha256(native_mask.numpy().tobytes()).hexdigest()
                    token_occupancy = occupancy_from_mask_logits(
                        native_mask, memory_grid_shape, device=memory_features.device
                    )
                    pool_kernel = [
                        int(native_mask.shape[0]) // memory_grid_shape[0],
                        int(native_mask.shape[1]) // memory_grid_shape[1],
                    ]
                    metadata = {
                        "frame": source_frame,
                        "stored_mask_field": mask_field,
                        "stored_mask_sha256": mask_sha256,
                        "stored_mask_shape": list(native_mask.shape),
                        "memory_grid_shape": list(memory_grid_shape),
                        "occupancy_pool_kernel": pool_kernel,
                        "occupancy_tokens": int(token_occupancy.numel()),
                    }
                    self.mask_cache[source_frame] = (token_occupancy, metadata)
                else:
                    token_occupancy, metadata = cached
                    if tuple(metadata["memory_grid_shape"]) != memory_grid_shape:
                        raise RuntimeError("A stored frame's memory grid changed between reads")
                    token_occupancy = token_occupancy.to(device=memory_features.device)
                if token_occupancy.numel() != int(span["stop"]) - int(span["start"]):
                    raise RuntimeError("Stored mask grid does not align with memory spatial tokens")
                maps.append(token_occupancy)
                self.mask_sources.append(dict(metadata))
            self.occupancy = torch.cat(maps)[None]
        return self.original_prepare(*args, **kwargs)

    def _sdpa(self, query, key, value, *args, **kwargs):
        if self.inside or self.active is None:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        if args or kwargs.get("attn_mask") is not None or kwargs.get("dropout_p", 0.0) != 0.0:
            raise RuntimeError("Unsupported SAM2 memory SDPA options")
        if kwargs.get("is_causal", False):
            raise RuntimeError("Causal SDPA is not expected in SAM2 memory attention")
        if self.occupancy is None:
            raise RuntimeError("Stored-mask occupancy was not prepared for this memory read")
        layer, pointers = self.active
        key_reduced, value_reduced, mass_bias, audit = compress_memory_kv(
            key,
            value,
            self.occupancy.to(key.device),
            self.spans,
            mode=self.method_mode,
            pointers=pointers,
            role_weight=self.role_weight,
        )
        self.inside = True
        try:
            output = self.original_sdpa(
                query,
                key_reduced,
                value_reduced,
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
            "layer": layer,
            "mask_sources": list(self.mask_sources),
            **audit,
        })
        return output
