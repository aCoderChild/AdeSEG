"""MCTF-style, attention-aware value consolidation for frozen SAM2 reads.

This adapts the official MCTF matching operator to SAM2 cross-attention. It
changes only past spatial values; native keys, pointers, queries, SAM2 weights,
memory writes, presence handling, and frame schedule remain untouched. Tokens
are merged at a fixed per-window count and restored to their source slots, so
this is an accuracy hypothesis, not an efficiency claim.
"""

from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import torch

from modeling.attention_distortion_observer import _sha256_tensor
from modeling.native_value_consolidation import NativeValueConsolidation


ROOT = Path(__file__).resolve().parents[1]
MCTF_REVISION = "2004da0cad3231fa74cbd60b6042b0bd133aacde"
MCTF_SOURCE = "utils/mctf.py"
DEFAULT_WINDOW = 4
DEFAULT_MERGES_PER_WINDOW = 4
DEFAULT_TAU_SIM = 1
DEFAULT_TAU_INFO = 20
DEFAULT_TAU_SIZE = 40
DEFAULT_BIDIRECTION = 1


@lru_cache(maxsize=1)
def author_mctf():
    """Load the unchanged, pinned official MCTF matching/merge implementation."""
    repo = ROOT / "external/MCTF"
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != MCTF_REVISION:
        raise RuntimeError("MCTF checkout differs from the recorded author revision")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", MCTF_SOURCE],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    source = repo / MCTF_SOURCE
    spec = importlib.util.spec_from_file_location("_adeseg_author_mctf", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the pinned MCTF source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mean_attention_over_queries(query, key, chunk_size=128, scale=None):
    """Compute each memory key's mean incoming attention in bounded Q chunks."""
    if query.ndim != 4 or key.ndim != 4 or query.shape[:2] != key.shape[:2]:
        raise ValueError("Expected aligned [B,H,Q,D] queries and [B,H,K,D] keys")
    if query.shape[-1] != key.shape[-1] or query.shape[0] != 1:
        raise ValueError("Expected one-object SAM2 queries/keys with matching head width")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    scale = float(scale) if scale is not None else query.shape[-1] ** -0.5
    key32 = key.float()
    total = torch.zeros(
        (query.shape[0], query.shape[1], key.shape[-2]),
        device=key.device,
        dtype=torch.float64,
    )
    for start in range(0, query.shape[-2], chunk_size):
        q = query[..., start:start + chunk_size, :].float()
        probability = torch.softmax(torch.matmul(q, key32.transpose(-2, -1)) * scale, dim=-1)
        total += probability.sum(dim=-2, dtype=torch.float64)
    return (total / query.shape[-2]).to(dtype=torch.float64)


def consolidate_attention_aware_values(
    query,
    keys,
    values,
    spans,
    *,
    pointers,
    merge_count=DEFAULT_MERGES_PER_WINDOW,
    window_size=DEFAULT_WINDOW,
    importance=True,
    chunk_size=128,
    tau_sim=DEFAULT_TAU_SIM,
    tau_info=DEFAULT_TAU_INFO,
    tau_size=DEFAULT_TAU_SIZE,
    bidirection=DEFAULT_BIDIRECTION,
):
    """Apply official MCTF matching to value-only merges in each past-frame tile.

    ``importance=False`` is the matched MCTF control: the same operator, merge
    count, pooling, and feature similarity, with uniform attention importance.
    """
    if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1:
        raise ValueError("Expected one-object aligned [B,H,K,D] memory keys/values")
    if query.shape[0] != keys.shape[0] or query.shape[1] != keys.shape[1]:
        raise ValueError("Query and memory batch/head counts differ")
    if not torch.isfinite(query).all() or not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("Q/K/V must be finite")
    if type(pointers) is not int or not 0 <= pointers < keys.shape[-2]:
        raise ValueError("Invalid SAM2 object-pointer tail length")
    if type(window_size) is not int or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if type(merge_count) is not int or not 0 <= merge_count <= window_size * window_size // 2:
        raise ValueError("merge_count must be between zero and half a window")
    if merge_count == 0:
        return keys, values, {"mode": "mctf", "merged_tokens": 0, "windows": 0}

    spatial = keys.shape[-2] - pointers
    if int(query.shape[-1]) != int(keys.shape[-1]):
        raise ValueError("SAM2 query/key head width mismatch")
    mean_attention = (
        _mean_attention_over_queries(query, keys)
        if importance else None
    )
    mctf = author_mctf()
    changed_values = values.clone()
    cursor = 0
    windows = 0
    merged = 0
    frame_audit = []
    for span in spans:
        start, stop = int(span["start"]), int(span["stop"])
        side = math.isqrt(stop - start)
        if start != cursor or side * side != stop - start:
            raise ValueError("Memory spans must be contiguous square spatial maps")
        if side % window_size:
            raise ValueError("Memory maps must be evenly tiled by the selected window")
        cursor = stop
        frame_windows = 0
        # Keep the prompt-conditioning map exact; only historical maps are adapted.
        if int(span["t_pos"]) == 0:
            frame_audit.append({"frame": int(span["frame"]), "merged_tokens": 0, "windows": 0})
            continue
        grid = torch.arange(start, stop, device=keys.device).reshape(side, side)
        for top in range(0, side, window_size):
            for left in range(0, side, window_size):
                indices = grid[top:top + window_size, left:left + window_size].flatten()
                local_key = keys.index_select(-2, indices)
                metric = local_key.float().mean(dim=1)
                norms = metric.norm(dim=-1)
                if not torch.isfinite(norms).all() or (norms <= 0).any():
                    raise ValueError("MCTF matching metrics must have finite nonzero norms")
                if importance:
                    # MCTF inverts mean incoming attention. Clamp only at the
                    # dtype's representable floor to avoid reciprocal overflow
                    # on numerically underflowed softmax entries.
                    local_attention = mean_attention.index_select(-1, indices).to(torch.float32)
                    local_attention = local_attention.unsqueeze(-2)
                    local_attention = local_attention.clamp_min(torch.finfo(local_attention.dtype).tiny)
                else:
                    local_attention = torch.ones(
                        (1, keys.shape[1], 1, indices.numel()),
                        device=keys.device,
                        dtype=torch.float32,
                    )
                size = torch.ones((1, indices.numel(), 1), device=keys.device, dtype=torch.float32)
                merge, _ = mctf.bipartite_soft_matching(
                    metric,
                    local_attention,
                    size,
                    r=merge_count,
                    num_protected=0,
                    tau_sim=tau_sim,
                    tau_info=tau_info if importance else 0,
                    tau_size=tau_size,
                    bidirection=bidirection,
                )
                # MCTF's ViT path keeps the reduced sequence; SAM2 needs its
                # dense spatial slots. Use the author's source-tracing operator
                # to map each original token to its merged group and restore
                # that group's value at the original coordinates.
                source_map = mctf.merge_source(merge, metric)
                if source_map.shape[0] != 1 or source_map.shape[-1] != indices.numel():
                    raise RuntimeError("MCTF source map does not match the original window")
                if not torch.all(source_map.sum(dim=1) == 1):
                    raise RuntimeError("MCTF source map must assign each token to one merged group")
                original_to_group = source_map[0].argmax(dim=0)
                pooled_heads = []
                pooled_mass = None
                local_values = values.index_select(-2, indices)
                for head in range(values.shape[1]):
                    pooled, mass, _ = mctf.merge_wavg(
                        merge,
                        local_values[:, head],
                        size=size,
                    )
                    if pooled.shape[-2] != source_map.shape[1]:
                        raise RuntimeError("MCTF pooled values and source map have different groups")
                    restored = pooled.index_select(-2, original_to_group)
                    pooled_heads.append(restored)
                    if pooled_mass is None:
                        pooled_mass = mass
                    elif not torch.equal(pooled_mass, mass):
                        raise RuntimeError("MCTF produced inconsistent token sizes across heads")
                changed_values.index_copy_(-2, indices, torch.stack(pooled_heads, dim=1))
                windows += 1
                frame_windows += 1
                merged += merge_count
        frame_audit.append({
            "frame": int(span["frame"]),
            "t_pos": int(span["t_pos"]),
            "merged_tokens": frame_windows * merge_count,
            "windows": frame_windows,
        })
    if cursor != spatial:
        raise ValueError("Memory spans do not cover exactly all spatial tokens")
    if pointers and not torch.equal(changed_values[..., -pointers:, :], values[..., -pointers:, :]):
        raise RuntimeError("MCTF changed object pointers")
    if not torch.isfinite(changed_values).all():
        raise RuntimeError("MCTF produced non-finite memory values")
    return keys, changed_values, {
        "mode": "mctf_attention_aware" if importance else "mctf_uniform_control",
        "spatial_keys": spatial,
        "pointer_keys": pointers,
        "merged_tokens": merged,
        "windows": windows,
        "frames": frame_audit,
    }


class MCTFMemoryValue(NativeValueConsolidation):
    """Read-only MCTF value adapter using current-to-memory attention importance."""

    def __init__(self, model, *, importance=True, merge_count=DEFAULT_MERGES_PER_WINDOW):
        super().__init__(model, mode="noop")
        self.importance = bool(importance)
        self.merge_count = int(merge_count)

    def _sdpa(self, query, key, value, *args, **kwargs):
        if self.inside or self.active is None:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        if args or kwargs.get("attn_mask") is not None or kwargs.get("dropout_p", 0) != 0:
            raise RuntimeError("MCTF adapter supports the native unmasked SAM2 SDPA call only")
        if kwargs.get("is_causal", False) or kwargs.get("scale") is not None or kwargs.get("enable_gqa", False):
            raise RuntimeError("Unsupported SAM2 SDPA option for the frozen MCTF adapter")
        layer, pointers = self.active
        before = [_sha256_tensor(x) for x in (query, key, value)]
        keys, values, audit = consolidate_attention_aware_values(
            query,
            key,
            value,
            self.spans,
            pointers=pointers,
            merge_count=self.merge_count,
            importance=self.importance,
        )
        if not torch.equal(keys, key):
            raise RuntimeError("Attention-aware value MCTF changed native keys")
        output = self.original_sdpa(query, keys, values, *args, **kwargs)
        if before != [_sha256_tensor(x) for x in (query, key, value)]:
            raise RuntimeError("MCTF mutated native SDPA inputs")
        self.calls.append({
            "frame": self.frame,
            "layer": layer,
            "qkv_input_sha256": before,
            "key_unchanged": True,
            "importance_enabled": self.importance,
            "source_frames": [
                {"frame": int(span["frame"]), "t_pos": int(span["t_pos"]),
                 "tokens": int(span["stop"] - span["start"])}
                for span in self.spans
            ],
            **audit,
        })
        return output
