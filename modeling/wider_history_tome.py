"""Fixed-budget ToMe reads over a wider SAM2 memory history.

The upstream ToMe matching and weighted merge are reused. Compression happens
after SAM2's K/V projections and RoPE, immediately before SDPA.
"""

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F

from modeling.region_constrained_tome import NativeRegionConstrainedToMe, author_tome


MAP_TOKENS = 64 * 64
NATIVE_NONCOND_MAPS = 6
MAX_NONCOND_MAPS = 12
NATIVE_NONCOND_BUDGET = MAP_TOKENS * NATIVE_NONCOND_MAPS
EXTENDED_NUM_MASKMEM = MAX_NONCOND_MAPS + 1


def allocate_history_budget(
    spans, *, map_tokens=MAP_TOKENS,
    native_noncond_maps=NATIVE_NONCOND_MAPS, max_noncond_maps=MAX_NONCOND_MAPS,
    allocation="uniform",
):
    """Allocate the native history budget uniformly or toward recent maps."""
    if allocation not in ("uniform", "recent_dense"):
        raise ValueError("Expected uniform or recent_dense history allocation")
    if not spans:
        raise ValueError("Expected selected memory spans")
    cond = [i for i, span in enumerate(spans) if int(span["t_pos"]) == 0]
    noncond = [i for i, span in enumerate(spans) if int(span["t_pos"]) > 0]
    if len(cond) != 1:
        raise ValueError("The PolypGen protocol expects exactly one conditioning map")
    if len(noncond) > max_noncond_maps:
        raise ValueError(f"At most {max_noncond_maps} non-conditioning maps are supported")

    quotas = {index: int(spans[index]["stop"]) - int(spans[index]["start"])
              for index in cond + noncond}
    count = len(noncond)
    if count <= native_noncond_maps:
        return quotas

    noncond_budget = map_tokens * native_noncond_maps
    newest_first = sorted(noncond, key=lambda index: int(spans[index]["t_pos"]), reverse=True)
    if allocation == "uniform":
        base, remainder = divmod(noncond_budget, count)
        for rank, index in enumerate(newest_first):
            quotas[index] = base + int(rank < remainder)
    else:
        # Preserve the six most recent maps at higher spatial density. Older
        # maps are retained at ToMe's 50% local-window limit; no map is dropped.
        oldest_count = count - native_noncond_maps
        oldest_first = list(reversed(newest_first))
        for index in oldest_first[:oldest_count]:
            quotas[index] = map_tokens // 2
        recent_budget = noncond_budget - oldest_count * (map_tokens // 2)
        base, remainder = divmod(recent_budget, native_noncond_maps)
        for rank, index in enumerate(newest_first[:native_noncond_maps]):
            quotas[index] = base + int(rank < remainder)
    if sum(quotas[index] for index in noncond) != noncond_budget:
        raise RuntimeError("Non-conditioning token quotas do not conserve the native budget")
    return quotas


def _budgeted_window_merge(keys, values, roles, merge_count, mode, tome):
    """Merge one local window with the pinned ToMe assignment and size weighting."""
    if merge_count == 0:
        mass = keys.new_ones((keys.shape[-2],), dtype=torch.float32)
        return keys, values, mass
    if merge_count < 0 or merge_count > keys.shape[-2] // 2:
        raise ValueError("Local ToMe merge count exceeds the upstream 50% limit")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("Memory K/V must be finite before ToMe")

    metric = keys.float().mean(dim=1)
    feature_norm = metric.norm(dim=-1, keepdim=True)
    if not torch.isfinite(feature_norm).all() or (feature_norm <= 0).any():
        raise ValueError("ToMe feature metrics must have finite nonzero norms")
    metric = F.normalize(metric, dim=-1)
    if mode == "regional":
        if roles.shape != (keys.shape[0], keys.shape[-2], 3):
            raise ValueError("Regional roles must align with the local memory window")
        role_norm = roles.float().norm(dim=-1, keepdim=True)
        if not torch.isfinite(role_norm).all() or (role_norm <= 0).any():
            raise ValueError("Predicted soft-role metrics must have finite nonzero norms")
        role_metric = F.normalize(roles.float(), dim=-1)
        metric = torch.cat((metric / math.sqrt(2.0), role_metric / math.sqrt(2.0)), dim=-1)
    elif mode != "global":
        raise ValueError("Expected regional or global ToMe matching")

    merge, _ = tome.bipartite_soft_matching(metric, int(merge_count))
    merged_keys, merged_values, masses = [], [], []
    for head in range(keys.shape[1]):
        key_out, key_size = tome.merge_wavg(merge, keys[:, head])
        value_out, value_size = tome.merge_wavg(merge, values[:, head])
        if not torch.equal(key_size, value_size):
            raise RuntimeError("ToMe assigned different mass to K and V")
        current_mass = key_size[0, :, 0].float()
        if masses and not torch.equal(masses[0], current_mass):
            raise RuntimeError("ToMe assignments differ across attention heads")
        masses.append(current_mass)
        merged_keys.append(key_out)
        merged_values.append(value_out)
    return (
        torch.stack(merged_keys, dim=1),
        torch.stack(merged_values, dim=1),
        masses[0],
    )


def compress_history_kv(
    keys, values, roles, spans, *, mode, pointers, window_size=4,
    native_map_tokens=MAP_TOKENS,
    native_noncond_maps=NATIVE_NONCOND_MAPS,
    max_noncond_maps=MAX_NONCOND_MAPS,
    allocation="uniform",
):
    """Compress maps to exact quotas while preserving per-frame age and pointers."""
    if mode not in ("regional", "global"):
        raise ValueError("Expected regional or global ToMe matching")
    if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1:
        raise ValueError("Expected matching single-object [1, heads, keys, channels] K/V")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("Memory K/V must be finite")
    if pointers < 0 or pointers >= keys.shape[-2]:
        raise ValueError("Invalid object-pointer count")
    spatial = keys.shape[-2] - pointers
    if roles.shape != (1, spatial, 3) or not torch.isfinite(roles).all():
        raise ValueError("Predicted soft roles do not match the spatial K/V grid")
    if (roles < 0).any() or not torch.allclose(
        roles.sum(-1), torch.ones_like(roles[..., 0]), atol=1e-5
    ):
        raise ValueError("Predicted soft roles must be nonnegative and sum to one")
    if window_size <= 0:
        raise ValueError("Window size must be positive")

    quotas = allocate_history_budget(
        spans, map_tokens=native_map_tokens,
        native_noncond_maps=native_noncond_maps,
        max_noncond_maps=max_noncond_maps,
        allocation=allocation,
    )
    cursor = 0
    for index, span in enumerate(spans):
        start, stop = int(span["start"]), int(span["stop"])
        side = math.isqrt(stop - start)
        if start != cursor or side * side != stop - start or side % window_size:
            raise ValueError("Memory spans must be contiguous square grids tiled by the window")
        cursor = stop
        if int(span["t_pos"]) == 0 and (stop - start) != native_map_tokens:
            raise ValueError("The conditioning map must retain its native 64x64 grid")
        if quotas[index] <= 0 or quotas[index] > stop - start:
            raise ValueError("Invalid per-frame token quota")
    if cursor != spatial:
        raise ValueError("Memory spans do not cover all spatial keys")

    if all(quotas[index] == int(span["stop"]) - int(span["start"])
           for index, span in enumerate(spans)):
        audit = {
            "mode": mode,
            "spatial_keys_before": spatial,
            "spatial_keys_after": spatial,
            "pointer_keys": pointers,
            "spatial_key_mass_sum": float(spatial),
            "key_mass_sum": float(spatial + pointers),
            "frames": [
                {
                    "frame": int(span["frame"]),
                    "t_pos": int(span["t_pos"]),
                    "source_tokens": int(span["stop"] - span["start"]),
                    "target_tokens": int(quotas[index]),
                    "merged_tokens": 0,
                    "windows": [],
                }
                for index, span in enumerate(spans)
            ],
        }
        bias = keys.new_zeros((1, 1, 1, keys.shape[-2]), dtype=torch.float32)
        return keys, values, bias, audit

    tome = author_tome()
    reduced_keys, reduced_values, masses = [], [], []
    frame_audit = []
    window_global = torch.arange(spatial, device=keys.device)
    role_maps = roles[0]
    for span_index, span in enumerate(spans):
        start, stop = int(span["start"]), int(span["stop"])
        side = math.isqrt(stop - start)
        source_count = stop - start
        target_count = quotas[span_index]
        merge_count = source_count - target_count
        windows = (side // window_size) ** 2
        base, remainder = divmod(merge_count, windows)
        window_audit = []
        local_window_index = 0
        for top in range(0, side, window_size):
            for left in range(0, side, window_size):
                local = window_global[start:stop].reshape(side, side)[
                    top:top + window_size, left:left + window_size
                ].reshape(-1)
                # Spread the remainder through the grid instead of concentrating
                # the extra merges in the first row-major windows.
                local_merges = base + int(
                    ((local_window_index + 1) * remainder) // windows
                    > (local_window_index * remainder) // windows
                )
                if local_merges > local.numel() // 2:
                    raise ValueError("Requested frame quota exceeds the local ToMe capacity")
                local_keys = keys.index_select(-2, local)
                local_values = values.index_select(-2, local)
                local_roles = role_maps.index_select(0, local)
                key_out, value_out, mass = _budgeted_window_merge(
                    local_keys, local_values, local_roles[None], local_merges, mode, tome
                )
                reduced_keys.append(key_out)
                reduced_values.append(value_out)
                masses.append(mass)
                window_audit.append({
                    "window": local_window_index,
                    "source_tokens": int(local.numel()),
                    "merged_tokens": int(local_merges),
                    "retained_tokens": int(local.numel()) - int(local_merges),
                })
                local_window_index += 1
        frame_audit.append({
            "frame": int(span["frame"]),
            "t_pos": int(span["t_pos"]),
            "source_tokens": source_count,
            "target_tokens": int(target_count),
            "merged_tokens": int(merge_count),
            "windows": window_audit,
        })

    spatial_keys = torch.cat(reduced_keys, dim=-2)
    spatial_values = torch.cat(reduced_values, dim=-2)
    spatial_mass = torch.cat(masses).to(device=keys.device, dtype=torch.float32)
    pointer_keys = keys[..., spatial:, :]
    pointer_values = values[..., spatial:, :]
    out_keys = torch.cat((spatial_keys, pointer_keys), dim=-2)
    out_values = torch.cat((spatial_values, pointer_values), dim=-2)
    all_mass = torch.cat((spatial_mass, spatial_mass.new_ones(pointers)))
    if pointers and (
        not torch.equal(out_keys[..., -pointers:, :], pointer_keys)
        or not torch.equal(out_values[..., -pointers:, :], pointer_values)
    ):
        raise RuntimeError("ToMe modified native object-pointer K/V")
    if int(spatial_keys.shape[-2]) != sum(quotas.values()):
        raise RuntimeError("Compressed token count differs from the exact allocated budget")
    if not torch.allclose(spatial_mass.sum(), torch.tensor(float(spatial), device=keys.device)):
        raise RuntimeError("ToMe cluster mass does not conserve source-token count")
    if int(all_mass.numel()) != out_keys.shape[-2] or not torch.isfinite(all_mass).all():
        raise RuntimeError("Compressed token mass is invalid")

    audit = {
        "mode": mode,
        "spatial_keys_before": spatial,
        "spatial_keys_after": int(spatial_keys.shape[-2]),
        "pointer_keys": pointers,
        "spatial_key_mass_sum": float(spatial_mass.sum()),
        "key_mass_sum": float(all_mass.sum()),
        "frames": frame_audit,
    }
    return out_keys, out_values, all_mass.log().view(1, 1, 1, -1), audit


class FixedBudgetLongHistoryToMe(NativeRegionConstrainedToMe):
    """Read a bounded history at the native SDPA budget with frozen SAM2 weights."""

    def __init__(self, model, mode="regional", window_size=4,
                 max_noncond_maps=MAX_NONCOND_MAPS, allocation="uniform"):
        if model.num_maskmem != NATIVE_NONCOND_MAPS + 1:
            raise ValueError("Expected native SAM2 num_maskmem=7 before entering the adapter")
        if (
            max_noncond_maps < NATIVE_NONCOND_MAPS
            or max_noncond_maps > MAX_NONCOND_MAPS
            or allocation not in ("uniform", "recent_dense")
        ):
            raise ValueError("History cap or allocation strategy is unsupported")
        super().__init__(model, mode=mode, window_size=window_size)
        self.max_noncond_maps = int(max_noncond_maps)
        self.allocation = allocation
        self._original_num_maskmem = None
        self._original_tpos_data = None

    def _compress_memory(self, keys, values, memberships, spans, pointers, merge_plan):
        if merge_plan is not None:
            raise ValueError("This study uses a fixed, formula-derived exact token budget")
        return compress_history_kv(
            keys, values, memberships, spans,
            mode=self.mode, pointers=pointers, window_size=self.window_size,
            max_noncond_maps=self.max_noncond_maps,
            allocation=self.allocation,
        )

    def _install_extended_time_slots(self):
        model = self.model
        if self._original_num_maskmem is not None:
            raise RuntimeError("Extended temporal slots are already installed")
        original = model.maskmem_tpos_enc.data
        if original.shape[0] != NATIVE_NONCOND_MAPS + 1:
            raise ValueError("Unexpected native temporal-slot table shape")
        # Index 0..5 encode ages 1..6; indices 6..11 reuse age 6; index 12
        # preserves the original conditioning-frame embedding.
        extended = torch.cat((
            original[:NATIVE_NONCOND_MAPS],
            original[NATIVE_NONCOND_MAPS - 1:NATIVE_NONCOND_MAPS].expand(
                self.max_noncond_maps - NATIVE_NONCOND_MAPS, *original.shape[1:]
            ),
            original[NATIVE_NONCOND_MAPS:NATIVE_NONCOND_MAPS + 1],
        ), dim=0)
        expected_rows = self.max_noncond_maps + 1
        if extended.shape[0] != expected_rows:
            raise RuntimeError("Extended temporal-slot table has an invalid length")
        self._original_num_maskmem = model.num_maskmem
        self._original_tpos_data = original
        model.maskmem_tpos_enc.data = extended
        model.num_maskmem = expected_rows

    def _restore_native_time_slots(self):
        if self._original_num_maskmem is None:
            return
        self.model.maskmem_tpos_enc.data = self._original_tpos_data
        self.model.num_maskmem = self._original_num_maskmem
        self._original_num_maskmem = None
        self._original_tpos_data = None

    def __enter__(self):
        self._install_extended_time_slots()
        try:
            return super().__enter__()
        except BaseException:
            self._restore_native_time_slots()
            raise

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self._restore_native_time_slots()
