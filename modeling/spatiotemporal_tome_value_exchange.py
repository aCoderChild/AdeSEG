"""Cross-frame value sharing with the official video-ToMe merge/unmerge.

This inference-only adapter matches memory tokens inside each 4x4 spatial tile
across SAM2's already-selected past maps, averages matched V vectors, and
unmerges them to the original token slots. Q, post-RoPE K, all positions,
pointer tokens, the memory bank and SAM2 weights remain unchanged.
"""

from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import torch

from modeling.attention_distortion_observer import _sha256_tensor
from modeling.region_constrained_tome import NativeRegionConstrainedToMe


ROOT = Path(__file__).resolve().parents[1]
VIDEO_TOME_REVISION = "267fb4915c0dcb55f87f555fd72dbb6df4a618b7"
VIDEO_TOME_SOURCE = "tome/merge.py"
VIDEO_TOME_SOURCE_SHA256 = "b70f6d48dfac425027174ccb857cf0768dd415cbc65e14466ea27dc73d22af57"
WINDOW_SIZE = 4
MERGE_DIVISOR = 4


@lru_cache(maxsize=1)
def author_video_tome():
    """Load the unmodified official video-ToMe matcher and merge operators."""
    repo = ROOT / "external/video-token-merging"
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != VIDEO_TOME_REVISION:
        raise RuntimeError("Video-ToMe checkout differs from its pinned revision")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", VIDEO_TOME_SOURCE],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    source = repo / VIDEO_TOME_SOURCE
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != VIDEO_TOME_SOURCE_SHA256:
        raise RuntimeError("Official video-ToMe merge source hash differs")
    spec = importlib.util.spec_from_file_location(
        "_aadeseg_author_video_tome_merge", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the pinned video-ToMe source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _memory_layout(keys, values, spans, pointers):
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("Expected aligned [B,H,N,D] memory keys and values")
    if keys.shape[0] < 1 or not 0 <= pointers < keys.shape[-2]:
        raise ValueError("Invalid memory batch or pointer count")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("Memory keys and values must be finite")
    spatial_count = keys.shape[-2] - pointers
    cursor = 0
    side = None
    history = []
    for span in spans:
        start, stop = int(span["start"]), int(span["stop"])
        count = stop - start
        grid_side = math.isqrt(count)
        if start != cursor or count <= 0 or grid_side * grid_side != count:
            raise ValueError("Memory spans must be contiguous square spatial maps")
        if side is None:
            side = grid_side
        elif side != grid_side:
            raise ValueError("All memory maps must share one spatial grid")
        cursor = stop
        if int(span["t_pos"]) > 0:
            history.append({**span, "start": start, "stop": stop, "side": grid_side})
    if cursor != spatial_count:
        raise ValueError("Memory spans do not cover exactly the spatial K/V prefix")
    if side is None:
        raise ValueError("At least one spatial memory map is required")
    return side, spatial_count, history


def _frame_audit(spans):
    return [
        {
            "frame": int(span["frame"]),
            "t_pos": int(span["t_pos"]),
            "source_tokens": int(span["stop"] - span["start"]),
            "target_tokens": int(span["stop"] - span["start"]),
            "merged_tokens": 0,
            "windows": [],
        }
        for span in spans
    ]


def _weighted_unmerge(keys, values, indices, frame_labels, merge_count, tome):
    selected_keys = keys.index_select(-2, indices)
    selected_values = values.index_select(-2, indices)
    metric = selected_keys.float().mean(dim=1)
    norms = metric.norm(dim=-1)
    if not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Video-ToMe matching keys must have finite nonzero norms")
    if indices.numel() % MERGE_DIVISOR or not 0 < merge_count <= indices.numel() // 2:
        raise ValueError("Invalid fixed ToMe merge count")

    merge, unmerge = tome.bipartite_soft_matching(metric, int(merge_count))
    labels = frame_labels.to(device=keys.device, dtype=torch.int64).view(1, -1, 1)
    labels = labels.expand(keys.shape[0], -1, -1)
    earliest = merge(labels, mode="amin")
    latest = merge(labels, mode="amax")
    cross_frame_groups = int((earliest != latest).sum().item())

    restored = []
    for head in range(values.shape[1]):
        pooled, _ = tome.merge_wavg(merge, selected_values[:, head])
        restored.append(unmerge(pooled))
    restored_values = torch.stack(restored, dim=1)
    return restored_values, cross_frame_groups


def exchange_memory_values(
    keys,
    values,
    spans,
    *,
    pointers,
    mode="cross_frame",
    window_size=WINDOW_SIZE,
):
    """Apply matched ToMe value averaging, restoring every original slot.

    ``local`` is the matched generic control: each historical frame is merged
    independently in its own 4x4 windows. ``cross_frame`` orders tokens in each
    tile by spatial cell then native memory slot, allowing the official
    bipartite matcher to group similar content across past frames. Both modes
    average exactly one quarter of their eligible tokens and unmerge to the
    same dense token count. The conditioning map (t_pos == 0) and pointer tail
    are untouched. With only one historical map, cross-frame falls back to the
    local control because no temporal pair exists.
    """
    if mode not in ("noop", "local", "cross_frame"):
        raise ValueError("Expected noop, local or cross_frame mode")
    if type(window_size) is not int or window_size <= 0:
        raise ValueError("Window size must be a positive integer")
    side, spatial_count, history = _memory_layout(keys, values, spans, pointers)
    if side % window_size:
        raise ValueError("Window size must evenly tile each memory grid")
    if mode == "noop" or not history:
        return keys, values, {
            "mode": mode,
            "scope": "noop",
            "eligible_frames": [int(x["frame"]) for x in history],
            "eligible_t_pos": [int(x["t_pos"]) for x in history],
            "eligible_tokens": 0,
            "merged_pairs": 0,
            "cross_frame_groups": 0,
            "frames": _frame_audit(spans),
            "spatial_keys_before": spatial_count,
            "spatial_keys_after": spatial_count,
            "spatial_key_mass_sum": float(spatial_count),
            "key_mass_sum": float(keys.shape[-2]),
            "pointer_keys": pointers,
        }

    tome = author_video_tome()
    changed_values = values.clone()
    merged_pairs = 0
    cross_frame_groups = 0
    tile_count = 0
    multi_frame = len(history) > 1
    scope = "cross_frame" if mode == "cross_frame" and multi_frame else "local"

    if scope == "local":
        for span in history:
            grid = torch.arange(
                span["start"], span["stop"], device=keys.device
            ).reshape(side, side)
            for top in range(0, side, window_size):
                for left in range(0, side, window_size):
                    indices = grid[
                        top : top + window_size, left : left + window_size
                    ].reshape(-1)
                    labels = torch.full(
                        (indices.numel(),), int(span["t_pos"]), device=keys.device
                    )
                    count = indices.numel() // MERGE_DIVISOR
                    restored, cross = _weighted_unmerge(
                        keys, values, indices, labels, count, tome
                    )
                    changed_values.index_copy_(-2, indices, restored)
                    merged_pairs += count
                    cross_frame_groups += cross
                    tile_count += 1
    else:
        for top in range(0, side, window_size):
            for left in range(0, side, window_size):
                ordered_indices = []
                ordered_labels = []
                for dy in range(window_size):
                    for dx in range(window_size):
                        for span in history:
                            ordered_indices.append(
                                span["start"]
                                + (top + dy) * side
                                + left
                                + dx
                            )
                            ordered_labels.append(int(span["t_pos"]))
                indices = torch.tensor(ordered_indices, device=keys.device)
                labels = torch.tensor(ordered_labels, device=keys.device)
                count = indices.numel() // MERGE_DIVISOR
                restored, cross = _weighted_unmerge(
                    keys, values, indices, labels, count, tome
                )
                changed_values.index_copy_(-2, indices, restored)
                merged_pairs += count
                cross_frame_groups += cross
                tile_count += 1

    if changed_values.shape[-2] != keys.shape[-2]:
        raise RuntimeError("Value unmerge changed the native token count")
    if not torch.equal(values[..., : history[0]["start"], :],
                       changed_values[..., : history[0]["start"], :]):
        raise RuntimeError("Value exchange changed a conditioning-map prefix")
    if pointers and not torch.equal(
        changed_values[..., -pointers:, :], values[..., -pointers:, :]
    ):
        raise RuntimeError("Value exchange changed object-pointer tokens")

    return keys, changed_values, {
        "mode": mode,
        "scope": scope,
        "eligible_frames": [int(x["frame"]) for x in history],
        "eligible_t_pos": [int(x["t_pos"]) for x in history],
        "eligible_tokens": sum(x["stop"] - x["start"] for x in history),
        "merged_pairs": merged_pairs,
        "cross_frame_groups": cross_frame_groups,
        "tiles": tile_count,
        "merge_fraction": 1.0 / MERGE_DIVISOR,
        "frames": _frame_audit(spans),
        "spatial_keys_before": spatial_count,
        "spatial_keys_after": spatial_count,
        "spatial_key_mass_sum": float(spatial_count),
        "key_mass_sum": float(keys.shape[-2]),
        "pointer_keys": pointers,
    }


class NativeSpatiotemporalValueExchange(NativeRegionConstrainedToMe):
    """Apply local or cross-frame ToMe value sharing at native memory SDPA."""

    requires_roles = False

    def __init__(self, model, *, mode="cross_frame"):
        if mode not in ("noop", "local", "cross_frame"):
            raise ValueError("Expected noop, local or cross_frame mode")
        super().__init__(model, mode="global")
        self.exchange_mode = mode

    def _sdpa(self, query, key, value, *args, **kwargs):
        if self.inside or self.active is None:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        if (
            args
            or kwargs.get("attn_mask") is not None
            or kwargs.get("dropout_p", 0.0) != 0.0
            or kwargs.get("is_causal", False)
            or kwargs.get("scale") is not None
            or kwargs.get("enable_gqa", False)
        ):
            raise RuntimeError("Unsupported options in native SAM2 memory SDPA")

        layer, pointers = self.active
        input_hashes = [_sha256_tensor(x) for x in (query, key, value)]
        output_key, output_value, audit = exchange_memory_values(
            key,
            value,
            self.spans,
            pointers=pointers,
            mode=self.exchange_mode,
        )
        if not torch.equal(output_key, key):
            raise RuntimeError("Value exchange changed native memory keys")
        output = self.original_sdpa(query, output_key, output_value, *args, **kwargs)
        if input_hashes != [_sha256_tensor(x) for x in (query, key, value)]:
            raise RuntimeError("Value exchange mutated native Q/K/V inputs")
        self.calls.append(
            {
                "frame": self.frame,
                "object_id": self.current_object_id,
                "layer": layer,
                "qkv_input_sha256": input_hashes,
                "key_unchanged": True,
                **audit,
            }
        )
        return output

