from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modeling.wider_history_tome import (
    FixedBudgetLongHistoryToMe,
    allocate_history_budget,
    compress_history_kv,
)


def make_spans(noncond, map_tokens=16):
    spans = [{"frame": 0, "t_pos": 0, "start": 0, "stop": map_tokens}]
    cursor = map_tokens
    # Oldest first, matching SAM2's memory concatenation order.
    for age in range(noncond, 0, -1):
        t_pos = 13 - age
        spans.append({
            "frame": noncond - age + 1,
            "t_pos": t_pos,
            "start": cursor,
            "stop": cursor + map_tokens,
        })
        cursor += map_tokens
    return spans


def test_history_budget_is_native_until_six_maps_then_exact_and_recent_favored():
    six = make_spans(6)
    six_quotas = allocate_history_budget(six, map_tokens=16)
    assert list(six_quotas.values()) == [16] * 7

    seven = make_spans(7)
    seven_quotas = allocate_history_budget(seven, map_tokens=16)
    assert sum(seven_quotas.values()) == 16 + 6 * 16
    noncond = seven[1:]
    assert [seven_quotas[index] for index in range(1, 8)] == [13, 13, 14, 14, 14, 14, 14]

    twelve = make_spans(12)
    twelve_quotas = allocate_history_budget(twelve, map_tokens=16)
    assert list(twelve_quotas.values()) == [16] + [8] * 12


@pytest.mark.parametrize("noncond", [7, 8, 9, 10])
def test_recent_dense_history_budget_keeps_all_maps_and_favors_newer_maps(noncond):
    map_tokens = 16
    spans = make_spans(noncond, map_tokens)
    quotas = allocate_history_budget(
        spans,
        map_tokens=map_tokens,
        max_noncond_maps=10,
        allocation="recent_dense",
    )
    values = [quotas[index] for index in range(1, len(spans))]
    oldest_count = noncond - 6
    assert sum(values) == map_tokens * 6
    assert values[:oldest_count] == [map_tokens // 2] * oldest_count
    assert all(value >= map_tokens // 2 for value in values)
    assert all(value <= map_tokens for value in values)
    assert values[oldest_count] >= max(values[:oldest_count], default=0)
    assert values[-1] == max(values[oldest_count:])


def test_recent_dense_compression_conserves_attention_mass_and_local_limit():
    torch.manual_seed(37)
    noncond, map_tokens, pointers = 10, 16, 2
    spans = make_spans(noncond, map_tokens)
    spatial = len(spans) * map_tokens
    keys = torch.randn(1, 2, spatial + pointers, 8)
    values = torch.randn_like(keys)
    roles = torch.softmax(torch.randn(1, spatial, 3), dim=-1)
    out_keys, out_values, bias, audit = compress_history_kv(
        keys,
        values,
        roles,
        spans,
        mode="global",
        pointers=pointers,
        window_size=2,
        native_map_tokens=map_tokens,
        native_noncond_maps=6,
        max_noncond_maps=10,
        allocation="recent_dense",
    )
    quotas = [row["target_tokens"] for row in audit["frames"]]
    assert quotas == [16, 8, 8, 8, 8, 10, 10, 11, 11, 11, 11]
    assert out_keys.shape == out_values.shape == (1, 2, 16 + 96 + pointers, 8)
    assert audit["spatial_keys_after"] == 112
    assert audit["key_mass_sum"] == float(spatial + pointers)
    assert torch.isfinite(bias).all()
    assert all(window["merged_tokens"] <= 2 for row in audit["frames"] for window in row["windows"])


def test_exact_budget_compression_matches_generic_and_regional_and_preserves_pointers():
    torch.manual_seed(31)
    noncond, map_tokens, pointers = 7, 16, 2
    spans = make_spans(noncond, map_tokens)
    spatial = len(spans) * map_tokens
    keys = torch.randn(1, 2, spatial + pointers, 8)
    values = torch.randn_like(keys)
    roles = torch.softmax(torch.randn(1, spatial, 3), dim=-1)
    pointer_keys, pointer_values = keys[..., -pointers:, :].clone(), values[..., -pointers:, :].clone()

    common = dict(
        spans=spans,
        mode="global",
        pointers=pointers,
        window_size=2,
        native_map_tokens=map_tokens,
        native_noncond_maps=6,
        max_noncond_maps=12,
    )
    generic = compress_history_kv(keys, values, roles, **common)
    regional = compress_history_kv(keys, values, roles, **{**common, "mode": "regional"})

    for compressed in (generic, regional):
        out_keys, out_values, bias, audit = compressed
        assert out_keys.shape == out_values.shape == (1, 2, map_tokens + 6 * map_tokens + pointers, 8)
        assert audit["spatial_keys_after"] == map_tokens * 7
        assert audit["spatial_key_mass_sum"] == float(spatial)
        assert audit["key_mass_sum"] == float(spatial + pointers)
        assert torch.equal(out_keys[..., -pointers:, :], pointer_keys)
        assert torch.equal(out_values[..., -pointers:, :], pointer_values)
        assert torch.isfinite(bias).all()
        assert sum(frame["target_tokens"] for frame in audit["frames"]) == map_tokens * 7
        assert all(frame["target_tokens"] == 16 for frame in audit["frames"][:1])

    assert generic[0].shape == regional[0].shape
    assert [frame["target_tokens"] for frame in generic[3]["frames"]] == [
        frame["target_tokens"] for frame in regional[3]["frames"]
    ]


def test_rejects_more_than_the_declared_history_limit():
    spans = make_spans(12)
    spans.append({"frame": 13, "t_pos": 13, "start": 13 * 16, "stop": 14 * 16})
    with pytest.raises(ValueError, match="At most 12"):
        allocate_history_budget(spans)


def test_cluster_mass_preserves_attention_when_merged_key_values_are_identical():
    torch.manual_seed(8)
    noncond, map_tokens, heads, channels = 7, 16, 2, 8
    spans = make_spans(noncond, map_tokens)
    spatial = len(spans) * map_tokens
    key_vectors = torch.randn(1, heads, 1, channels)
    value_vectors = torch.randn(1, heads, 1, channels)
    keys = key_vectors.expand(1, heads, spatial, channels).clone()
    values = value_vectors.expand(1, heads, spatial, channels).clone()
    roles = torch.full((1, spatial, 3), 1 / 3)
    pointers = 0

    out_keys, out_values, bias, audit = compress_history_kv(
        keys,
        values,
        roles,
        spans,
        mode="regional",
        pointers=pointers,
        window_size=2,
        native_map_tokens=map_tokens,
        native_noncond_maps=6,
        max_noncond_maps=12,
    )
    query = torch.randn(1, heads, 5, channels)
    dense = torch.nn.functional.scaled_dot_product_attention(query, keys, values)
    corrected = torch.nn.functional.scaled_dot_product_attention(
        query, out_keys, out_values, attn_mask=bias
    )

    assert audit["spatial_keys_after"] < spatial
    assert torch.allclose(corrected, dense, atol=1e-6, rtol=1e-6)


def test_extended_temporal_slots_repeat_oldest_and_restore_original_parameter():
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_maskmem = 7
            self.maskmem_tpos_enc = nn.Parameter(torch.arange(14.0).reshape(7, 1, 1, 2))
            self.memory_attention = SimpleNamespace(layers=[])

        def _prepare_memory_conditioned_features(self, frame_idx):
            return frame_idx

    model = FakeModel()
    original = model.maskmem_tpos_enc.detach().clone()
    adapter = FixedBudgetLongHistoryToMe(model)
    adapter._install_extended_time_slots()
    expanded = model.maskmem_tpos_enc.detach().clone()
    assert model.num_maskmem == 13
    assert expanded.shape == (13, 1, 1, 2)
    assert torch.equal(expanded[:6], original[:6])
    assert torch.equal(expanded[6:12], original[5:6].expand(6, 1, 1, 2))
    assert torch.equal(expanded[12:13], original[6:7])
    for age in range(1, 7):
        native_t_pos = 7 - age
        extended_t_pos = 13 - age
        native_index = 7 - native_t_pos - 1
        extended_index = 13 - extended_t_pos - 1
        assert torch.equal(expanded[extended_index], original[native_index])
    for age in range(7, 13):
        extended_t_pos = 13 - age
        extended_index = 13 - extended_t_pos - 1
        assert torch.equal(expanded[extended_index], original[5])
    adapter._restore_native_time_slots()
    assert model.num_maskmem == 7
    assert torch.equal(model.maskmem_tpos_enc.detach(), original)


def test_ten_map_temporal_slots_repeat_oldest_and_restore_original_parameter():
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_maskmem = 7
            self.maskmem_tpos_enc = nn.Parameter(torch.arange(14.0).reshape(7, 1, 1, 2))
            self.memory_attention = SimpleNamespace(layers=[])

        def _prepare_memory_conditioned_features(self, frame_idx):
            return frame_idx

    model = FakeModel()
    original = model.maskmem_tpos_enc.detach().clone()
    adapter = FixedBudgetLongHistoryToMe(model, max_noncond_maps=10)
    adapter._install_extended_time_slots()
    expanded = model.maskmem_tpos_enc.detach().clone()
    assert model.num_maskmem == 11
    assert expanded.shape == (11, 1, 1, 2)
    assert torch.equal(expanded[:6], original[:6])
    assert torch.equal(expanded[6:10], original[5:6].expand(4, 1, 1, 2))
    assert torch.equal(expanded[10:11], original[6:7])
    for age in range(1, 7):
        native_t_pos = 7 - age
        extended_t_pos = 11 - age
        native_index = 7 - native_t_pos - 1
        extended_index = 11 - extended_t_pos - 1
        assert torch.equal(expanded[extended_index], original[native_index])
    for age in range(7, 11):
        extended_t_pos = 11 - age
        extended_index = 11 - extended_t_pos - 1
        assert torch.equal(expanded[extended_index], original[5])
    adapter._restore_native_time_slots()
    assert model.num_maskmem == 7
    assert torch.equal(model.maskmem_tpos_enc.detach(), original)
