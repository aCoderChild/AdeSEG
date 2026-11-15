"""Position-aware ToMe compression for SAM2 memory attention.

This adaptation reuses the pinned upstream ToMe matching/weighted-merge
operations and SAM2's native projections and RoPE helper. It compresses
``memory`` before key projection/RoPE, then applies each merged key's
representative spatial coordinate explicitly. Position embeddings can follow
the original weighted-mean rule or be kept from the representative token.
Object-pointer tokens remain at the end of the sequence and pass through.

The module is an experimental integration, not an established improvement.
"""

from functools import lru_cache
import hashlib
import importlib.util
import inspect
import json
import math
import copy
from pathlib import Path
import subprocess
import sys
from types import MethodType

import torch
import torch.nn.functional as F

from modeling.prediction_conditioned_affinity import memory_role_sequence


ROOT = Path(__file__).resolve().parents[1]
TOME_REVISION = "af95e4b1befa172dadccd8c81e223b10090f9579"
TOME_SOURCE = "tome/merge.py"
WINDOW_SIZE = 4


def _tensor_sha256(tensor):
    """Hash a detached contiguous tensor on the frozen CPU FP32 runtime."""
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _compact_frame_provenance(frame_audit):
    """Keep a compact partition certificate for every memory frame/window."""
    compact = []
    for frame in frame_audit:
        side = math.isqrt(int(frame["input_spatial_keys"]))
        window_size = int(frame["window_size"])
        windows_per_side = side // window_size
        windows = []
        for window in frame["windows"]:
            clusters = window.pop("clusters")
            members = [
                int(index)
                for cluster in clusters
                for index in cluster["original_grid_indices"]
            ]
            window_index = int(window["window"])
            top = (window_index // windows_per_side) * window_size
            left = (window_index % windows_per_side) * window_size
            expected_members = sorted(
                (top + dy) * side + left + dx
                for dy in range(window_size)
                for dx in range(window_size)
            )
            if (
                sorted(members) != expected_members
                or len(set(members)) != len(members)
                or any(
                    int(cluster["mass"]) != len(cluster["original_grid_indices"])
                    or int(cluster["representative_grid_index"])
                    not in cluster["original_grid_indices"]
                    for cluster in clusters
                )
            ):
                raise RuntimeError("Stored ToMe clusters do not partition their source window")
            encoded = json.dumps(clusters, sort_keys=True, separators=(",", ":")).encode()
            windows.append({
                **window,
                "cluster_count": len(clusters),
                "partition_sha256": hashlib.sha256(encoded).hexdigest(),
                "partition_valid": True,
            })
        compact.append({**frame, "windows": windows})
    return compact


@lru_cache(maxsize=1)
def author_tome():
    """Load unmodified matching and weighted aggregation from official ToMe."""
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
    spec = importlib.util.spec_from_file_location("_aadeseg_positioned_tome", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the pinned ToMe source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _merge_group(memory, position, metric, grid_positions, count, *, position_mode):
    """Merge memory tokens and retain each cluster's destination coordinate.

    Inputs use batch-first token layout ``[1, N, C]``. The destination token
    is the representative coordinate, matching the official bipartite merge's
    source-to-destination assignment. Position embeddings are either merged
    by weighted mean or copied from the representative token.
    """
    count = int(count)
    if position_mode not in ("mean", "representative"):
        raise ValueError("Expected position_mode='mean' or 'representative'")
    if count <= 0:
        size = memory.new_ones((1, memory.shape[1], 1), dtype=torch.float32)
        members = [[int(position)] for position in grid_positions.tolist()]
        return memory, position, size[0, :, 0], grid_positions, members

    tome = author_tome()
    merge, _ = tome.bipartite_soft_matching(metric, count)
    token_sizes = memory.new_ones((1, memory.shape[1], 1), dtype=torch.float32)
    merged_memory, sizes = tome.merge_wavg(merge, memory, token_sizes)
    if position_mode == "mean":
        merged_position, position_sizes = tome.merge_wavg(merge, position, token_sizes)
        if not torch.equal(sizes, position_sizes):
            raise RuntimeError("ToMe produced different memory/position cluster sizes")

    # Official ToMe orders unmatched A tokens first and B destinations last.
    # Merging an identity source map recovers the exact output-cluster layout.
    identity = torch.eye(memory.shape[1], device=memory.device)[None]
    source_map = tome.merge_source(merge, identity)[0]
    if not torch.equal(
        source_map.sum(dim=0),
        torch.ones(memory.shape[1], device=memory.device, dtype=source_map.dtype),
    ):
        raise RuntimeError("ToMe source map is not a partition of the input tokens")
    source_sizes = source_map.sum(dim=-1)
    if not torch.equal(source_sizes, sizes[0, :, 0].to(source_sizes.dtype)):
        raise RuntimeError("ToMe source membership differs from weighted-merge mass")
    a_count = (memory.shape[1] + 1) // 2
    unmatched_a = a_count - count
    representative_indices = []
    for output_index in range(source_map.shape[0]):
        if output_index < unmatched_a:
            members = torch.nonzero(source_map[output_index], as_tuple=False).flatten()
            if members.numel() != 1:
                raise RuntimeError("Unexpected ToMe unmatched-source cluster")
            representative_indices.append(members[0])
        else:
            destination_b = output_index - unmatched_a
            destination = 2 * destination_b + 1
            if destination >= memory.shape[1]:
                raise RuntimeError("ToMe destination index is outside the source group")
            representative_indices.append(torch.tensor(destination, device=memory.device))
    representative_indices = torch.stack(representative_indices).long()
    representatives = grid_positions.index_select(0, representative_indices)
    if position_mode == "representative":
        merged_position = position.index_select(1, representative_indices)
    member_groups = []
    for source_row in source_map:
        member_indices = torch.nonzero(source_row, as_tuple=False).flatten()
        member_groups.append(
            [int(index) for index in grid_positions.index_select(0, member_indices).tolist()]
        )
    return merged_memory, merged_position, sizes[0, :, 0], representatives, member_groups


def build_matching_metric(attention, memory, position, *, mode="key"):
    """Build a ToMe similarity metric from frozen SAM2 projections.

    ``key`` matches the existing position-consistent candidate. ``key_value``
    gives normalized, head-averaged K and V equal weight, so tokens with similar
    attention logits but dissimilar retrieved content are less likely to merge.
    Both modes reuse the pinned ToMe bipartite matcher; this changes only its
    metric, not SAM2's projections or attention computation.
    """
    if mode not in ("key", "key_value"):
        raise ValueError("Expected matching mode 'key' or 'key_value'")
    key_projection = attention.k_proj(memory + position)
    key_heads = attention._separate_heads(key_projection, attention.num_heads)
    key_metric = key_heads.float().mean(dim=1)
    if not torch.isfinite(key_metric).all():
        raise ValueError("ToMe key matching metrics must be finite")
    key_norms = key_metric.norm(dim=-1)
    if (key_norms <= 0).any():
        raise ValueError("ToMe key matching metrics must be nonzero")
    if mode == "key":
        return key_metric

    value_projection = attention.v_proj(memory)
    value_heads = attention._separate_heads(value_projection, attention.num_heads)
    value_metric = value_heads.float().mean(dim=1)
    if not torch.isfinite(value_metric).all():
        raise ValueError("ToMe value matching metrics must be finite")

    # Normalize each semantic role separately before concatenation. The
    # official ToMe matcher normalizes the concatenated vector again, so the
    # two halves contribute equally regardless of projection scale.
    key_unit = F.normalize(key_metric, dim=-1)
    value_unit = F.normalize(value_metric, dim=-1)
    metric = torch.cat((key_unit, value_unit), dim=-1)
    metric_norms = metric.norm(dim=-1)
    if not torch.isfinite(metric_norms).all() or (metric_norms <= 0).any():
        raise ValueError("Joint K/V matching metrics must be finite and nonzero")
    return metric


def compress_memory_inputs(memory, position, memberships, spans, *, mode,
                           pointers, window_size=WINDOW_SIZE, merge_plan=None,
                           attention=None, hash_tensors=False,
                           position_mode="mean", matching_mode="key"):
    """Merge spatial memory before projection/RoPE; preserve pointer tail.

    ``memory`` and ``position`` have batch-first layout ``[1,S,C]`` inside
    ``MemoryAttentionLayer._forward_ca`` (SAM2 transposes before layer calls).
    Matching metrics are computed from the native frozen K projection before
    RoPE. The same ToMe assignment is applied to memory and position. Memory
    content is mass-averaged; ``position_mode`` chooses whether positional
    embeddings are averaged or copied from each cluster's representative.
    """
    if mode not in ("regional", "global", "no_merge"):
        raise ValueError("Expected regional, global, or no_merge mode")
    if matching_mode not in ("key", "key_value"):
        raise ValueError("Expected matching_mode='key' or 'key_value'")
    if position_mode not in ("mean", "representative"):
        raise ValueError("Expected position_mode='mean' or 'representative'")
    if memory.ndim != 3 or memory.shape != position.shape or memory.shape[0] != 1:
        raise ValueError("Expected matching batch-first memory/position [1,S,C]")
    if not torch.isfinite(memory).all() or not torch.isfinite(position).all():
        raise ValueError("Memory and position inputs must be finite")
    if pointers < 0 or pointers >= memory.shape[1]:
        raise ValueError("Invalid native pointer count")
    if attention is None or not hasattr(attention, "k_proj"):
        raise ValueError("The native SAM2 cross-attention module is required")
    if window_size <= 0:
        raise ValueError("Window size must be positive")

    spatial = memory.shape[1] - pointers
    if memberships.shape != (1, spatial, 3):
        raise ValueError("Predicted region roles do not match the spatial memory tokens")
    if not torch.isfinite(memberships).all() or (memberships < 0).any():
        raise ValueError("Region memberships must be finite and nonnegative")
    if not torch.allclose(
        memberships.sum(-1), torch.ones_like(memberships[..., 0]), atol=1e-5
    ):
        raise ValueError("Region memberships must sum to one")

    # Matching uses native frozen projections before RoPE. The joint option
    # additionally accounts for values, which carry the content returned by
    # memory attention and are not represented by key similarity alone.
    metric = build_matching_metric(
        attention, memory[:, :spatial], position[:, :spatial], mode=matching_mode
    )
    metric_norms = metric.norm(dim=-1)
    if not torch.isfinite(metric_norms).all() or (metric_norms <= 0).any():
        raise ValueError("ToMe matching metrics must be finite and nonzero")
    role_ids = memberships.argmax(dim=-1)[0]

    merged_memory, merged_position, merged_masses, rope_positions = [], [], [], []
    frame_audit = []
    cursor = 0
    for span in spans:
        start, stop = int(span["start"]), int(span["stop"])
        side = math.isqrt(stop - start)
        if start != cursor or side * side != stop - start or side % window_size:
            raise ValueError("Memory spans must be contiguous square grids tiled by windows")
        cursor = stop
        local_memory = memory[:, start:stop]
        local_position = position[:, start:stop]
        local_metric = metric[:, start:stop]
        local_roles = role_ids[start:stop]
        grid_indices = torch.arange(stop - start, device=memory.device).reshape(side, side)
        frame_output_memory, frame_output_position, frame_masses, frame_rope = [], [], [], []
        windows = []
        window_index = 0
        frame_merges = 0

        for top in range(0, side, window_size):
            for left in range(0, side, window_size):
                grid_positions = grid_indices[
                    top:top + window_size, left:left + window_size
                ].reshape(-1)
                roles = local_roles.index_select(0, grid_positions)
                groups = [grid_positions[roles == region] for region in range(3)]
                regional_counts = [int(group.numel()) // 2 for group in groups]
                target_merges = sum(regional_counts)
                planned = None
                if merge_plan is not None:
                    frame_plan = merge_plan.get(span["frame"])
                    if frame_plan is None or window_index not in frame_plan:
                        raise ValueError(
                            f"Missing matched token budget for {span['frame']} window {window_index}"
                        )
                    planned = int(frame_plan[window_index])
                    if planned < 0 or planned > grid_positions.numel() // 2:
                        raise ValueError("Matched ToMe merge count exceeds the local limit")
                if mode == "global":
                    if planned is None:
                        raise ValueError("Generic ToMe requires a frozen matched merge plan")
                    target_merges = planned
                elif mode == "no_merge":
                    target_merges = 0

                assigned = torch.cat(groups) if groups else grid_positions.new_empty((0,))
                if not torch.equal(torch.sort(assigned).values, torch.sort(grid_positions).values):
                    raise RuntimeError("Region groups do not partition their local window")

                window_audit = {
                    "window": window_index,
                    "input_keys": int(grid_positions.numel()),
                    "regional_budget": int(sum(regional_counts)),
                    "merges": int(target_merges),
                    "clusters": [],
                }
                windows.append(window_audit)
                window_index += 1
                frame_merges += target_merges

                if mode == "regional":
                    merge_groups = zip(groups, regional_counts, strict=True)
                elif mode == "global":
                    merge_groups = ((grid_positions, target_merges),)
                else:
                    merge_groups = ((grid_positions, 0),)

                for group_positions, count in merge_groups:
                    if not group_positions.numel():
                        continue
                    group_memory = local_memory.index_select(1, group_positions)
                    group_position = local_position.index_select(1, group_positions)
                    group_metric = local_metric.index_select(1, group_positions)
                    group_grid_positions = group_positions
                    out_memory, out_position, out_mass, out_rep, member_groups = _merge_group(
                        group_memory, group_position, group_metric,
                        group_grid_positions, count, position_mode=position_mode,
                    )
                    frame_output_memory.append(out_memory)
                    frame_output_position.append(out_position)
                    frame_masses.append(out_mass)
                    frame_rope.append(out_rep)
                    for representative, mass, members in zip(
                        out_rep.tolist(), out_mass.tolist(), member_groups, strict=True
                    ):
                        if int(mass) != len(members) or int(representative) not in members:
                            raise RuntimeError("Merged token provenance and mass are inconsistent")
                        window_audit["clusters"].append({
                            "source_frame": int(span["frame"]),
                            "t_pos": int(span["t_pos"]),
                            "window": int(window_index - 1),
                            "representative_grid_index": int(representative),
                            "representative_yx": [
                                int(representative) // side,
                                int(representative) % side,
                            ],
                            "original_grid_indices": members,
                            "original_yx": [[index // side, index % side] for index in members],
                            "mass": int(mass),
                        })
                window_audit["clusters"].sort(
                    key=lambda cluster: cluster["representative_grid_index"]
                )

        frame_memory = torch.cat(frame_output_memory, dim=1)
        frame_position = torch.cat(frame_output_position, dim=1)
        frame_mass = torch.cat(frame_masses)
        frame_rope_positions = torch.cat(frame_rope).long()
        if torch.unique(frame_rope_positions).numel() != frame_rope_positions.numel():
            raise RuntimeError("Merged spatial tokens have duplicate representative coordinates")
        spatial_order = torch.argsort(frame_rope_positions)
        merged_memory.append(frame_memory.index_select(1, spatial_order))
        merged_position.append(frame_position.index_select(1, spatial_order))
        merged_masses.append(frame_mass.index_select(0, spatial_order))
        rope_positions.append(frame_rope_positions.index_select(0, spatial_order))
        frame_audit.append({
            "frame": span["frame"],
            "t_pos": int(span["t_pos"]),
            "window_size": window_size,
            "input_spatial_keys": stop - start,
            "merged_keys": frame_merges,
            "output_spatial_keys": stop - start - frame_merges,
            "windows": windows,
        })
        if merge_plan is not None and len(merge_plan[span["frame"]]) != window_index:
            raise ValueError("Matched plan contains an unexpected number of windows")

    if cursor != spatial or not merged_memory:
        raise ValueError("Memory spans do not cover all spatial tokens")
    spatial_memory = torch.cat(merged_memory, dim=1)
    spatial_position = torch.cat(merged_position, dim=1)
    spatial_mass = torch.cat(merged_masses).to(device=memory.device, dtype=torch.float32)
    spatial_rope = torch.cat(rope_positions).long()
    pointer_memory = memory[:, spatial:]
    pointer_position = position[:, spatial:]
    out_memory = torch.cat((spatial_memory, pointer_memory), dim=1)
    out_position = torch.cat((spatial_position, pointer_position), dim=1)
    masses = torch.cat((spatial_mass, spatial_mass.new_ones(pointers)))

    if pointers and (
        not torch.equal(out_memory[:, -pointers:], memory[:, -pointers:])
        or not torch.equal(out_position[:, -pointers:], position[:, -pointers:])
    ):
        raise RuntimeError("ToMe modified native pointer inputs")
    if int(masses.numel()) != out_memory.shape[1] or not torch.isfinite(masses).all():
        raise RuntimeError("Compressed attention masses are invalid")
    if not torch.allclose(spatial_mass.sum(), spatial_mass.new_tensor(float(spatial)), atol=1e-5, rtol=0):
        raise RuntimeError("ToMe did not conserve original spatial-token mass")
    if mode == "no_merge" and (
        not torch.equal(spatial_memory, memory[:, :spatial])
        or not torch.equal(spatial_position, position[:, :spatial])
        or not torch.equal(spatial_rope, torch.cat([
            torch.arange(int(span["start"]), int(span["stop"]), device=memory.device)
            - int(span["start"]) for span in spans
        ]))
    ):
        raise RuntimeError("No-merge path did not preserve memory/position order exactly")

    audit = {
        "rope_positions": spatial_rope,
        "attention_bias": masses.log().view(1, 1, 1, -1),
        "pointers": pointers,
        "pointer_keys": pointers,
        "spatial_keys_before": spatial,
        "spatial_keys_after": int(spatial_mass.numel()),
        "spatial_key_mass_sum": float(spatial_mass.sum()),
        "key_mass_sum": float(masses.sum()),
        "frames": frame_audit,
        "position_mode": position_mode,
        "matching_mode": matching_mode,
        "no_merge_inputs_identity": bool(
            mode == "no_merge"
            and torch.equal(out_memory, memory)
            and torch.equal(out_position, position)
        ),
    }
    if hash_tensors:
        audit.update({
            "input_memory_sha256": _tensor_sha256(memory),
            "input_position_sha256": _tensor_sha256(position),
            "output_memory_sha256": _tensor_sha256(out_memory),
            "output_position_sha256": _tensor_sha256(out_position),
            "attention_bias_sha256": _tensor_sha256(audit["attention_bias"]),
            "rope_positions_sha256": _tensor_sha256(spatial_rope),
        })
    return out_memory, out_position, audit


def position_indexed_rope_attention(attention, q, k, v, *, pointers,
                                    rope_positions, attention_bias):
    """Run SAM2 RoPEAttention using explicit spatial locations for merged K."""
    from sam2.modeling.position_encoding import apply_rotary_enc

    q = attention.q_proj(q)
    k = attention.k_proj(k)
    v = attention.v_proj(v)
    q = attention._separate_heads(q, attention.num_heads)
    k = attention._separate_heads(k, attention.num_heads)
    v = attention._separate_heads(v, attention.num_heads)

    side = math.isqrt(q.shape[-2])
    if side * side != q.shape[-2]:
        raise ValueError("SAM2 memory queries must have a square spatial grid")
    attention.freqs_cis = attention.freqs_cis.to(q.device)
    if attention.freqs_cis.shape[0] != q.shape[-2]:
        attention.freqs_cis = attention.compute_cis(end_x=side, end_y=side).to(q.device)
    if pointers != k.shape[-2] - rope_positions.numel():
        raise ValueError("RoPE indices and pointer tail do not align with merged keys")
    if rope_positions.numel() and (
        rope_positions.min() < 0 or rope_positions.max() >= q.shape[-2]
    ):
        raise ValueError("A merged key has an invalid representative spatial coordinate")

    empty_q = q.new_empty((*q.shape[:-2], 0, q.shape[-1]))
    q_rotated, _ = apply_rotary_enc(q, empty_q, attention.freqs_cis)
    spatial_keys = k[:, :, :rope_positions.numel()]
    pointer_keys = k[:, :, rope_positions.numel():]
    if rope_positions.numel():
        key_freqs = attention.freqs_cis.index_select(0, rope_positions.to(q.device))
        empty_k = spatial_keys.new_empty((*spatial_keys.shape[:-2], 0, spatial_keys.shape[-1]))
        spatial_keys, _ = apply_rotary_enc(spatial_keys, empty_k, key_freqs)
    key_rotated = torch.cat((spatial_keys, pointer_keys), dim=-2)
    out = F.scaled_dot_product_attention(
        q_rotated, key_rotated, v,
        attn_mask=attention_bias.to(device=q.device, dtype=q.dtype),
        dropout_p=0.0,
    )
    out = attention._recombine_heads(out)
    return attention.out_proj(out)


class PositionConsistentRegionToMe:
    """Apply region-constrained ToMe before SAM2's native K projection/RoPE."""

    def __init__(self, model, mode="regional", window_size=WINDOW_SIZE,
                 merge_plan_by_target=None, object_id_by_output_dict=None,
                 passthrough=False, verify_no_merge_parity=False,
                 provenance_target=None, hash_tensors_target=None,
                 position_mode="mean", matching_mode="key"):
        if mode not in ("regional", "global", "no_merge"):
            raise ValueError("Expected regional, global, or no_merge mode")
        if verify_no_merge_parity and mode != "no_merge":
            raise ValueError("Native parity checks are available only in no_merge mode")
        if matching_mode not in ("key", "key_value"):
            raise ValueError("Expected matching_mode='key' or 'key_value'")
        self.model = model
        self.mode = mode
        self.window_size = int(window_size)
        if position_mode not in ("mean", "representative"):
            raise ValueError("Expected position_mode='mean' or 'representative'")
        self.position_mode = position_mode
        self.matching_mode = matching_mode
        self.merge_plan_by_target = merge_plan_by_target
        self.object_id_by_output_dict = object_id_by_output_dict
        self.passthrough = bool(passthrough)
        self.verify_no_merge_parity = bool(verify_no_merge_parity)
        self.provenance_target = provenance_target
        self.hash_tensors_target = hash_tensors_target
        self.provenance_reads = []
        self.memberships, self.spans, self.frame = None, [], None
        self.current_object_id = None
        self.pending = {}
        self.calls = []
        self.original_prepare = model._prepare_memory_conditioned_features
        self.prepare_signature = inspect.signature(self.original_prepare)
        self.original_attention_forwards = []
        self.original_layer_cross_attention = []

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
                self.memberships, self.spans = memory_role_sequence(
                    self.model, self.frame, data["output_dict"], data["num_frames"],
                    data["track_in_reverse"], "regional",
                )
            return self.original_prepare(*args, **kwargs)
        finally:
            self.current_object_id = previous_object_id

    def _plan_for_target(self):
        if self.merge_plan_by_target is None:
            return None
        key = self.frame
        if self.object_id_by_output_dict is not None:
            key = (self.frame, self.current_object_id)
        if key not in self.merge_plan_by_target:
            raise RuntimeError(f"Missing matched token budget for target {key}")
        return self.merge_plan_by_target[key]

    def _wrap_cross_attn(self, attention, layer_index, original_forward):
        manager = self

        def wrapped(this, q, k, v, num_k_exclude_rope=0):
            payload = manager.pending.get(layer_index)
            if payload is None:
                return original_forward(q, k, v, num_k_exclude_rope=num_k_exclude_rope)
            if int(num_k_exclude_rope) != int(payload["pointers"]):
                raise RuntimeError("SAM2 pointer count changed after memory compression")
            native_output = None
            if manager.verify_no_merge_parity:
                native_output = original_forward(
                    q, k, v, num_k_exclude_rope=num_k_exclude_rope
                )
            output = position_indexed_rope_attention(
                this, q, k, v,
                pointers=payload["pointers"],
                rope_positions=payload["rope_positions"],
                attention_bias=payload["attention_bias"],
            )
            manager.calls.append({
                "frame": manager.frame,
                "object_id": manager.current_object_id,
                "layer": layer_index,
                **payload["audit"],
            })
            if native_output is not None:
                max_abs = float((output - native_output).abs().max().detach().cpu())
                if not torch.allclose(output, native_output, atol=1e-6, rtol=1e-5):
                    raise RuntimeError(
                        f"No-merge native attention parity failed at layer {layer_index}: "
                        f"max_abs={max_abs}"
                    )
                manager.calls[-1]["native_parity_max_abs"] = max_abs
                if manager.hash_tensors_target == manager.frame:
                    manager.calls[-1]["native_output_sha256"] = _tensor_sha256(native_output)
                    manager.calls[-1]["candidate_output_sha256"] = _tensor_sha256(output)
            return output

        return MethodType(wrapped, attention)

    def _wrap_layer(self, layer, layer_index, original_ca):
        manager = self

        def wrapped(this, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
            if manager.passthrough or manager.memberships is None:
                manager.pending.pop(layer_index, None)
                manager.calls.append({
                    "frame": manager.frame,
                    "object_id": manager.current_object_id,
                    "layer": layer_index,
                    "mode": "passthrough",
                    "spatial_keys_before": int(memory.shape[1]) - int(num_k_exclude_rope),
                    "spatial_keys_after": int(memory.shape[1]) - int(num_k_exclude_rope),
                    "pointer_keys": int(num_k_exclude_rope),
                })
                return original_ca(
                    tgt, memory, query_pos, pos,
                    num_k_exclude_rope=num_k_exclude_rope,
                )

            plan = manager._plan_for_target()
            compressed_memory, compressed_pos, payload = compress_memory_inputs(
                memory, pos, manager.memberships.to(memory.device), manager.spans,
                mode=manager.mode,
                pointers=int(num_k_exclude_rope),
                window_size=manager.window_size,
                merge_plan=plan,
                attention=this.cross_attn_image,
                hash_tensors=manager.hash_tensors_target == manager.frame,
                position_mode=manager.position_mode,
                matching_mode=manager.matching_mode,
            )
            raw_frames = payload["frames"]
            if manager.provenance_target == manager.frame:
                manager.provenance_reads.append({
                    "frame": manager.frame,
                    "object_id": manager.current_object_id,
                    "layer": layer_index,
                    "frames": copy.deepcopy(raw_frames),
                })
            payload["audit"] = payload.copy()
            payload["audit"].pop("rope_positions", None)
            payload["audit"].pop("attention_bias", None)
            payload["audit"]["frames"] = _compact_frame_provenance(raw_frames)
            payload["audit"]["mode"] = manager.mode
            manager.pending[layer_index] = payload
            try:
                tgt2 = this.norm2(tgt)
                tgt2 = this.cross_attn_image(
                    q=tgt2 + query_pos if this.pos_enc_at_cross_attn_queries else tgt2,
                    k=compressed_memory + compressed_pos
                    if this.pos_enc_at_cross_attn_keys else compressed_memory,
                    v=compressed_memory,
                    num_k_exclude_rope=num_k_exclude_rope,
                )
                tgt = tgt + this.dropout2(tgt2)
                return tgt
            finally:
                manager.pending.pop(layer_index, None)

        return MethodType(wrapped, layer)

    def __enter__(self):
        if any(parameter.requires_grad for parameter in self.model.parameters()) or self.model.training:
            raise RuntimeError("SAM2 must be frozen and in evaluation mode")
        layers = self.model.memory_attention.layers
        if len(layers) != 4:
            raise RuntimeError("Expected all four native SAM2 memory-attention layers")
        if not all(layer.pos_enc_at_cross_attn_keys for layer in layers):
            raise RuntimeError("Unexpected SAM2 memory-key positional-encoding configuration")
        self.model._prepare_memory_conditioned_features = self._prepare
        try:
            for index, layer in enumerate(layers):
                attention = layer.cross_attn_image
                original_forward = attention.forward
                original_ca = layer._forward_ca
                self.original_attention_forwards.append((attention, original_forward))
                self.original_layer_cross_attention.append((layer, original_ca))
                attention.forward = self._wrap_cross_attn(attention, index, original_forward)
                layer._forward_ca = self._wrap_layer(layer, index, original_ca)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        self.model._prepare_memory_conditioned_features = self.original_prepare
        for attention, forward in self.original_attention_forwards:
            attention.forward = forward
        for layer, forward in self.original_layer_cross_attention:
            layer._forward_ca = forward
        self.pending.clear()
