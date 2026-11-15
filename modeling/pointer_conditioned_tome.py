"""Pointer-conditioned local matching around the pinned author ToMe merger.

Each stored spatial feature is matched using its native projected memory key
plus its cosine compatibility with that same frame's stored object pointer.
The cue is continuous: it changes ToMe pair similarity, not SAM2 attention
weights, object-presence decisions, frame selection, or memory writes.
"""

import math

import torch
import torch.nn.functional as F

from modeling.region_constrained_tome import (
    NativeRegionConstrainedToMe,
    _compress_group,
    WINDOW_SIZE,
)


def pointer_affinity_sequence(spans, outputs_by_frame):
    """Return same-frame spatial-feature/object-pointer cosine affinities [B,N]."""
    blocks = []
    batch_size = None
    for span in spans:
        frame = int(span["frame"])
        try:
            output = outputs_by_frame[frame]
        except KeyError as exc:
            raise ValueError(f"Missing stored output for memory frame {frame}") from exc
        features = output["maskmem_features"]
        pointer = output["obj_ptr"]
        if features.ndim != 4 or pointer.ndim != 2:
            raise ValueError("Expected memory features [B,C,H,W] and pointer [B,Cp]")
        batch, channels, height, width = features.shape
        if pointer.shape[0] != batch or pointer.shape[-1] % channels:
            raise ValueError("Pointer channels must split into aligned memory-width vectors")
        if batch_size is None:
            batch_size = batch
        if batch != batch_size:
            raise ValueError("Memory object batch changed across source frames")
        if not torch.isfinite(features).all() or not torch.isfinite(pointer).all():
            raise ValueError("Memory features and object pointers must be finite")

        spatial = features.flatten(2).transpose(1, 2).float()
        pointer_parts = pointer.to(device=features.device).float().reshape(batch, -1, channels)
        spatial = F.normalize(spatial, dim=-1)
        pointer_parts = F.normalize(pointer_parts, dim=-1)
        affinity = torch.einsum("bnc,brc->bnr", spatial, pointer_parts).mean(dim=-1)
        expected = int(span["stop"]) - int(span["start"])
        if affinity.shape[1] != expected:
            raise ValueError("Pointer affinity does not align with the memory span")
        blocks.append(affinity)
    if not blocks:
        raise ValueError("At least one selected spatial memory span is required")
    return torch.cat(blocks, dim=1)


def compress_pointer_conditioned_kv(
    keys,
    values,
    spans,
    affinities,
    *,
    pointers,
    window_size=WINDOW_SIZE,
    merge_divisor=2,
    merge_plan=None,
):
    """Apply local ToMe using same-frame pointer affinity as an added metric.

    Merge counts match generic ToMe when supplied the generic arm's ``merge_plan``.
    Spatial K/V means receive ToMe's conserved-mass attention bias; the pointer
    tail is appended unchanged. The saved spatial slots are not decoder outputs.
    """
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("Expected aligned [B,H,N,D] keys and values")
    batch, _, key_count, _ = keys.shape
    if not 0 <= pointers < key_count:
        raise ValueError("Invalid pointer-tail count")
    spatial = key_count - pointers
    if affinities.shape != (batch, spatial):
        raise ValueError("Pointer affinities must align with all spatial memory keys")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("Memory keys and values must be finite")
    if not torch.isfinite(affinities).all():
        raise ValueError("Pointer affinities must be finite")
    if not spans:
        raise ValueError("At least one selected spatial memory span is required")
    if window_size <= 0 or type(merge_divisor) is not int or merge_divisor < 2:
        raise ValueError("Invalid local ToMe window or merge divisor")

    key_metric = F.normalize(keys[..., :spatial, :].float().mean(dim=1), dim=-1)
    matching_metric = torch.cat((key_metric, affinities.float().unsqueeze(-1)), dim=-1)
    compressed_keys, compressed_values, masses = [], [], []
    frame_audit, cursor = [], 0
    for span in spans:
        start, stop = int(span["start"]), int(span["stop"])
        side = math.isqrt(stop - start)
        if start != cursor or side * side != stop - start or side % window_size:
            raise ValueError("Expected contiguous square memory grids tiled by the window")
        cursor = stop
        frame_plan = None if merge_plan is None else merge_plan.get(int(span["frame"]))
        if merge_plan is not None and frame_plan is None:
            raise ValueError(f"Missing matched merge plan for frame {span['frame']}")
        grid = torch.arange(start, stop, device=keys.device).reshape(side, side)
        frame_windows, window_idx = [], 0
        for top in range(0, side, window_size):
            for left in range(0, side, window_size):
                indices = grid[top:top + window_size, left:left + window_size].flatten()
                merges = int(indices.numel()) // merge_divisor
                if frame_plan is not None:
                    try:
                        merges = int(frame_plan[window_idx])
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing window {window_idx} for frame {span['frame']}"
                        ) from exc
                if not 0 <= merges <= indices.numel() // 2:
                    raise ValueError("Matched merge count exceeds the local ToMe limit")
                k_out, v_out, mass = _compress_group(
                    keys, values, indices, merges, matching_metric=matching_metric
                )
                compressed_keys.append(k_out)
                compressed_values.append(v_out)
                masses.append(mass)
                frame_windows.append({
                    "window": window_idx,
                    "original_spatial_keys": int(indices.numel()),
                    "merges": merges,
                })
                window_idx += 1
        if frame_plan is not None and len(frame_plan) != window_idx:
            raise ValueError(f"Unexpected window count for frame {span['frame']}")
        frame_affinity = affinities[:, start:stop]
        frame_audit.append({
            "frame": int(span["frame"]),
            "t_pos": int(span["t_pos"]),
            "original_spatial_keys": stop - start,
            "pointer_affinity_mean": float(frame_affinity.mean()),
            "pointer_affinity_std": float(frame_affinity.std(unbiased=False)),
            "windows": frame_windows,
        })
    if cursor != spatial:
        raise ValueError("Memory spans do not cover the spatial keys")

    spatial_keys = torch.cat(compressed_keys, dim=-2)
    spatial_values = torch.cat(compressed_values, dim=-2)
    spatial_mass = torch.cat(masses).to(device=keys.device, dtype=torch.float32)
    out_keys = torch.cat((spatial_keys, keys[..., spatial:, :]), dim=-2)
    out_values = torch.cat((spatial_values, values[..., spatial:, :]), dim=-2)
    all_mass = torch.cat((spatial_mass, spatial_mass.new_ones(pointers)))
    if not torch.allclose(spatial_mass.sum(), torch.tensor(float(spatial), device=keys.device),
                          atol=1e-5, rtol=0):
        raise RuntimeError("ToMe merge masses do not conserve spatial key count")
    if pointers and (
        not torch.equal(out_keys[..., -pointers:, :], keys[..., -pointers:, :])
        or not torch.equal(out_values[..., -pointers:, :], values[..., -pointers:, :])
    ):
        raise RuntimeError("Pointer-conditioned matching changed native pointer tokens")
    if all_mass.numel() != out_keys.shape[-2] or not torch.isfinite(all_mass).all():
        raise RuntimeError("Invalid attention mass after pointer-conditioned ToMe")
    bias = all_mass.log().view(1, 1, 1, -1)
    audit = {
        "mode": "pointer_conditioned",
        "spatial_keys_before": spatial,
        "spatial_keys_after": int(spatial_keys.shape[-2]),
        "pointer_keys": pointers,
        "spatial_key_mass_sum": float(spatial_mass.sum()),
        "key_mass_sum": float(all_mass.sum()),
        "frames": frame_audit,
    }
    if spatial_keys.shape[-2] == spatial:
        return keys, values, bias, audit
    return out_keys, out_values, bias, audit


class NativePointerConditionedToMe(NativeRegionConstrainedToMe):
    """Use same-frame memory/pointer affinity without computing region labels."""

    requires_roles = False

    def __init__(self, model, *, window_size=WINDOW_SIZE, merge_divisor=2,
                 merge_plan_by_target=None, object_id_by_output_dict=None,
                 passthrough=False):
        super().__init__(
            model, mode="global", window_size=window_size,
            merge_plan_by_target=merge_plan_by_target,
            object_id_by_output_dict=object_id_by_output_dict,
            passthrough=passthrough, merge_divisor=merge_divisor,
        )
        self.current_output_dict = None

    def _prepare(self, *args, **kwargs):
        bound = self.prepare_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        previous = self.current_output_dict
        self.current_output_dict = bound.arguments["output_dict"]
        try:
            return super()._prepare(*args, **kwargs)
        finally:
            self.current_output_dict = previous

    def _compress_memory(self, keys, values, memberships, spans, pointers, merge_plan):
        if self.current_output_dict is None:
            raise RuntimeError("Pointer-conditioned ToMe has no active source bank")
        outputs = {
            **self.current_output_dict["non_cond_frame_outputs"],
            **self.current_output_dict["cond_frame_outputs"],
        }
        affinities = pointer_affinity_sequence(spans, outputs).to(keys.device)
        return compress_pointer_conditioned_kv(
            keys, values, spans, affinities, pointers=pointers,
            window_size=self.window_size, merge_divisor=self.merge_divisor,
            merge_plan=merge_plan,
        )
