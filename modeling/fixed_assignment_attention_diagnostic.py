"""Measure existing ToMe clusters without changing a SAM2 attention read.

This is diagnostic glue around the pinned author merge/unmerge operations.
Counterfactual errors are coupled and must not be added as causal effects.
"""

import math

import torch
import torch.nn.functional as F

from modeling.attention_distortion_observer import _output_distortion, _sha256_tensor
from modeling.region_constrained_tome import author_tome
from modeling.wider_history_tome import compress_history_kv, FixedBudgetLongHistoryToMe


def expand_history_cluster_means(keys, values, roles, spans, *, pointers,
                                 allocation="uniform", max_noncond_maps=10,
                                 native_map_tokens=4096, window_size=4):
    """Re-expand the active generic merger's means to original member slots.

    Source frame/window quotas come from the existing merger. Matching is
    replayed once per window using the same metric and official operation;
    its unmerge restores member order, not the original token information.
    Object pointers remain untouched in every view.
    """
    compressed_key, compressed_value, bias, audit = compress_history_kv(
        keys, values, roles, spans, mode="global", pointers=pointers,
        allocation=allocation, max_noncond_maps=max_noncond_maps,
        native_map_tokens=native_map_tokens, window_size=window_size,
    )
    expanded_key, expanded_value = keys.clone(), values.clone()
    tome = author_tome()
    for span, frame in zip(spans, audit["frames"], strict=True):
        if frame["merged_tokens"] == 0:
            continue
        start, stop = int(span["start"]), int(span["stop"])
        side = math.isqrt(stop-start)
        grid = torch.arange(start, stop, device=keys.device).reshape(side, side)
        windows_per_row = side // window_size
        for window in frame["windows"]:
            count = int(window["merged_tokens"])
            if count == 0:
                continue
            top = (int(window["window"]) // windows_per_row) * window_size
            left = (int(window["window"]) % windows_per_row) * window_size
            indices = grid[top:top+window_size, left:left+window_size].flatten()
            local_key = keys.index_select(-2, indices)
            local_value = values.index_select(-2, indices)
            metric = F.normalize(local_key.float().mean(dim=1), dim=-1)
            merge, unmerge = tome.bipartite_soft_matching(metric, count)
            key_heads, value_heads = [], []
            for head in range(keys.shape[1]):
                mean_key, key_mass = tome.merge_wavg(merge, local_key[:, head])
                mean_value, value_mass = tome.merge_wavg(merge, local_value[:, head])
                if not torch.equal(key_mass, value_mass):
                    raise RuntimeError("Diagnostic K/V assignments have different masses")
                key_heads.append(unmerge(mean_key))
                value_heads.append(unmerge(mean_value))
            expanded_key.index_copy_(-2, indices, torch.stack(key_heads, dim=1))
            expanded_value.index_copy_(-2, indices, torch.stack(value_heads, dim=1))
    if pointers:
        for changed, original in ((expanded_key, keys), (expanded_value, values)):
            if not torch.equal(changed[..., -pointers:, :], original[..., -pointers:, :]):
                raise RuntimeError("Diagnostic changed an object-pointer token")
    return expanded_key, expanded_value, compressed_key, compressed_value, bias, audit


def measure_fixed_assignment_read(query, keys, values, roles, spans, *, pointers,
                                  allocation="uniform", max_noncond_maps=10,
                                  native_map_tokens=4096, window_size=4,
                                  sdpa=F.scaled_dot_product_attention,
                                  native_output=None):
    """Return scalar counterfactual measurements; never return altered model output.

    Supply the original native SDPA callable when used inside an observer so
    that the measurement cannot recursively invoke a patched SDPA function.
    Exact cluster identities hold in real arithmetic; the reported residual
    captures floating-point differences, not an independent V contribution.
    """
    ke, ve, km, vm, bias, audit = expand_history_cluster_means(
        keys, values, roles, spans, pointers=pointers, allocation=allocation,
        max_noncond_maps=max_noncond_maps, native_map_tokens=native_map_tokens,
        window_size=window_size,
    )
    native = native_output if native_output is not None else sdpa(query, keys, values)
    key_only = sdpa(query, ke, values)
    value_only = sdpa(query, keys, ve)
    joint_expanded = sdpa(query, ke, ve)
    mask = None if audit["spatial_keys_after"] == audit["spatial_keys_before"] else bias.to(
        device=query.device, dtype=query.dtype
    )
    joint_compressed = sdpa(query, km, vm, attn_mask=mask, dropout_p=0.0)
    outputs = {
        "key_mean_only_expanded": key_only,
        "value_mean_only_expanded": value_only,
        "joint_expanded": joint_expanded,
        "joint_compressed": joint_compressed,
    }
    if any(not torch.isfinite(out).all() for out in (native, *outputs.values())):
        raise RuntimeError("Diagnostic attention contains nonfinite output")
    # This residual must be interpreted against runtime-specific tolerances.
    identity = _output_distortion(key_only, joint_compressed)
    interaction = joint_expanded.float()-key_only.float()-value_only.float()+native.float()
    return {
        "scope": "fixed-query, fixed-bank generic ToMe diagnostic; no Dice inference",
        "counterfactuals": {name: _output_distortion(native, out)
                            for name, out in outputs.items()},
        "joint_vs_key_only_identity_residual": identity,
        "interaction_l2": float(interaction.norm()),
        "interpretation": "Counterfactual errors are coupled, not additive causal effects.",
        "memory_audit": audit,
    }


class FixedAssignmentSelectivityObserver(FixedBudgetLongHistoryToMe):
    """Measure shadows beside a dense ten-map read, returning its exact output.

    The original native bank supplies all selected maps. The dense ten-map
    reference itself changes read coverage; callers must also verify a native
    six-map replay and must never write any diagnostic result into that bank.
    This bounded wrapper currently supports the single-object PolypGen study.
    """

    def __init__(self, model, *, object_id=1):
        super().__init__(model, mode="global", max_noncond_maps=10, allocation="uniform")
        self.current_object_id = int(object_id)
        self.records = []

    def _sdpa(self, query, key, value, *args, **kwargs):
        if self.inside or self.active is None:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        if (
            args or kwargs.get("attn_mask") is not None
            or kwargs.get("dropout_p", 0.0) != 0.0
            or kwargs.get("is_causal", False)
            or kwargs.get("scale") is not None or kwargs.get("enable_gqa", False)
        ):
            raise RuntimeError("Unsupported memory SDPA options for the diagnostic")
        if self.memberships is None or not self.spans:
            raise RuntimeError("Diagnostic memory roles/source spans are missing")
        layer, pointers = self.active
        if any(r["frame"] == self.frame and r["layer"] == layer for r in self.records):
            raise RuntimeError("Diagnostic observed a target/layer more than once")
        native = self.original_sdpa(query, key, value, *args, **kwargs)
        originals = [x.detach().clone() for x in (query, key, value)]
        measurements = measure_fixed_assignment_read(
            query, key, value, self.memberships.to(key.device), self.spans,
            pointers=pointers, sdpa=self.original_sdpa, native_output=native,
        )
        for old, current in zip(originals, (query, key, value), strict=True):
            if not torch.equal(old, current):
                raise RuntimeError("Diagnostic changed a native Q/K/V input")
        residual = measurements["joint_vs_key_only_identity_residual"]
        relative_error = residual["relative_squared_error"]
        if (
            (relative_error is None and residual["max_absolute_difference"] != 0)
            or (relative_error is not None and relative_error > 1e-8)
            or residual["max_absolute_difference"] > 5e-5
        ):
            raise RuntimeError("Fixed-cluster identity exceeds frozen FP32 tolerance")
        self.records.append({
            "frame": self.frame, "object_id": self.current_object_id, "layer": layer,
            "q_sha256": _sha256_tensor(query), "k_sha256": _sha256_tensor(key),
            "v_sha256": _sha256_tensor(value),
            "dense_output_sha256": _sha256_tensor(native),
            "q_shape": list(query.shape), "k_shape": list(key.shape),
            "measurements": measurements,
        })
        return native

    def validate_target(self, frame):
        if [r["layer"] for r in self.records if r["frame"] == frame] != [0, 1, 2, 3]:
            raise RuntimeError("Diagnostic requires exactly four layers per target")
