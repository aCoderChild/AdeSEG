import torch

from modeling.mctf_memory_value import (
    consolidate_attention_aware_values,
    _mean_attention_over_queries,
)


def _fixture():
    torch.manual_seed(3205)
    query = torch.randn(1, 2, 16, 8)
    keys = torch.randn(1, 2, 34, 8)
    values = torch.randn_like(keys)
    spans = [
        {"frame": 0, "t_pos": 0, "start": 0, "stop": 16},
        {"frame": 1, "t_pos": 1, "start": 16, "stop": 32},
    ]
    return query, keys, values, spans


def test_attention_aware_mctf_changes_only_past_values_with_fixed_budget():
    query, keys, values, spans = _fixture()
    original_keys, original_values = keys.clone(), values.clone()
    out_keys, out_values, audit = consolidate_attention_aware_values(
        query, keys, values, spans, pointers=2
    )
    assert torch.equal(out_keys, keys)
    assert torch.equal(out_values[..., :16, :], values[..., :16, :])
    assert torch.equal(out_values[..., -2:, :], values[..., -2:, :])
    assert not torch.equal(out_values[..., 16:32, :], values[..., 16:32, :])
    assert out_values.shape == values.shape
    assert torch.isfinite(out_values).all()
    assert audit["merged_tokens"] == 4
    assert audit["windows"] == 1
    assert torch.equal(keys, original_keys) and torch.equal(values, original_values)


def test_uniform_attention_control_uses_same_mctf_budget_and_is_replayable():
    query, keys, values, spans = _fixture()
    first = consolidate_attention_aware_values(
        query, keys, values, spans, pointers=2, importance=False
    )
    second = consolidate_attention_aware_values(
        query, keys, values, spans, pointers=2, importance=False
    )
    assert first[2]["merged_tokens"] == second[2]["merged_tokens"] == 4
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_attention_importance_changes_fixed_similarity_merge_assignments():
    query, keys, values, spans = _fixture()
    aware = consolidate_attention_aware_values(
        query, keys, values, spans, pointers=2, importance=True
    )
    control = consolidate_attention_aware_values(
        query, keys, values, spans, pointers=2, importance=False
    )
    assert aware[2]["merged_tokens"] == control[2]["merged_tokens"] == 4
    assert torch.equal(aware[0], control[0])
    assert not torch.equal(aware[1], control[1])


def test_zero_query_has_uniform_incoming_attention_and_matches_control():
    _, keys, values, spans = _fixture()
    query = torch.zeros_like(_fixture()[0])
    importance = _mean_attention_over_queries(query, keys)
    assert torch.allclose(importance, importance[..., :1].expand_as(importance))
    aware = consolidate_attention_aware_values(
        query, keys, values, spans, pointers=2, importance=True
    )
    control = consolidate_attention_aware_values(
        query, keys, values, spans, pointers=2, importance=False
    )
    assert torch.equal(aware[1], control[1])


def test_zero_merge_budget_is_exact_identity():
    query, keys, values, spans = _fixture()
    out_keys, out_values, audit = consolidate_attention_aware_values(
        query, keys, values, spans, pointers=2, merge_count=0
    )
    assert out_keys is keys and out_values is values
    assert audit["merged_tokens"] == audit["windows"] == 0
