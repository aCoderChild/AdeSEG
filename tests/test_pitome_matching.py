import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from modeling.author_token_matching import use_matching
from modeling.pitome_matching import pitome_backend, use_pitome
from modeling.region_constrained_tome import compress_memory_kv
from scripts import run_pitome_causal_diagnostic as causal

PROTOCOL = json.loads((Path(__file__).resolve().parents[1] /
                       'protocols/pitome_protection_engineering_v1.json').read_text())
REPOSITORY = Path(PROTOCOL['pitome_repository'])


@pytest.mark.parametrize('tokens', range(1, 17))
def test_exact_budget_complete_assignments_and_weighted_values(tokens):
    backend = pitome_backend(REPOSITORY)
    generator = torch.Generator().manual_seed(tokens)
    metric = torch.randn(1, tokens, 7, generator=generator)
    original = torch.randn(1, tokens, 3, generator=generator)
    for count in range(tokens // 4 + 1):
        merge, _ = backend.bipartite_soft_matching(metric, count)
        values, mass = backend.merge_wavg(merge, original)
        assignments = backend.merge_source(merge, metric)
        assert values.shape == (1, tokens-count, 3)
        assert torch.equal(assignments.sum(1), torch.ones(1, tokens))
        assert torch.equal(assignments.sum(-1), mass[..., 0])
        torch.testing.assert_close(values, (assignments @ original) / mass)
        assert mass.sum() == tokens
        if count == 0:
            assert torch.equal(values, original)
        # Explicitly protected tokens are copied, regardless of value width.
        protected_count = tokens - 2 * count
        assert torch.equal(mass[:, :protected_count], torch.ones(1, protected_count, 1))


def test_distinct_outlier_remains_unmerged():
    backend = pitome_backend(REPOSITORY)
    metric = torch.ones(1, 16, 3)
    metric[:, -1] = -1
    merge, _ = backend.bipartite_soft_matching(metric, 4)
    assignments = backend.merge_source(merge, metric)
    row = assignments[0, :, -1].nonzero().item()
    assert row < 8  # Explicit protection prefix, not a destination by chance.
    assert assignments[0, row].sum() == 1


def test_quarter_budget_controls_share_heads_values_and_untouched_pointers():
    generator = torch.Generator().manual_seed(15)
    keys = torch.randn(1, 2, 18, 8, generator=generator)
    values = 2 * keys
    labels = torch.tensor([[0]*5 + [1]*5 + [2]*6])
    roles = F.one_hot(labels, 3).float()
    spans = [{'frame': 3, 't_pos': 0, 'start': 0, 'stop': 16}]
    budgets = []
    for context in (use_matching('tome'), use_pitome(REPOSITORY)):
        with context:
            for mode in ('regional', 'global'):
                k, v, bias, audit = compress_memory_kv(keys, values, roles, spans,
                    mode=mode, pointers=2, merge_divisor=4)
                torch.testing.assert_close(v, 2*k)
                assert torch.equal(k[..., -2:, :], keys[..., -2:, :])
                assert torch.equal(v[..., -2:, :], values[..., -2:, :])
                assert bias.exp().sum() == 18
                budgets.append(audit['frames'][0]['windows'])
    assert all(value == budgets[0] for value in budgets)
    assert budgets[0][0]['merges'] == 3


def test_rejects_half_budget_and_invalid_features():
    backend = pitome_backend(REPOSITORY)
    with pytest.raises(ValueError, match='quarter'):
        backend.bipartite_soft_matching(torch.ones(1, 16, 3), 8)
    with pytest.raises(ValueError, match='nonzero'):
        backend.bipartite_soft_matching(torch.zeros(1, 16, 3), 4)


def test_zero_removal_preserves_original_key_order():
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(1, 2, 18, 8, generator=generator)
    values = torch.randn(1, 2, 18, 8, generator=generator)
    roles = F.one_hot(torch.tensor([[0, 1, 2, 0]*4]), 3).float()
    spans = [{'frame': 3, 't_pos': 0, 'start': 0, 'stop': 16}]
    for mode in ('regional', 'global'):
        k, v, bias, audit = compress_memory_kv(keys, values, roles, spans,
            mode=mode, pointers=2, merge_divisor=4097)
        assert torch.equal(k, keys)
        assert torch.equal(v, values)
        assert not bias.count_nonzero()
        assert audit['spatial_keys_before'] == audit['spatial_keys_after']


def test_causal_wrapper_uses_quarter_budget_and_restores_reader(monkeypatch):
    seen = []
    original_manager = causal.reader.NativeRegionConstrainedToMe
    original_matching = causal.reader.use_matching

    def manager(*args, **kwargs):
        seen.append(kwargs['merge_divisor'])
        return 'quarter-budget manager'

    def read(*args, **kwargs):
        assert causal.reader.NativeRegionConstrainedToMe(None) == 'quarter-budget manager'
        with causal.reader.use_matching('pitome'):
            from modeling.region_constrained_tome import author_tome
            merge, _ = author_tome().bipartite_soft_matching(torch.ones(1, 16, 3), 4)
            assert merge(torch.ones(1, 16, 3)).shape[1] == 12
        return 'read completed'

    monkeypatch.setattr(causal, 'NativeRegionConstrainedToMe', manager)
    monkeypatch.setattr(causal.reader, 'test_read', read)
    assert causal.make_protected_reader(PROTOCOL)() == 'read completed'
    assert seen == [4]
    assert causal.reader.NativeRegionConstrainedToMe is original_manager
    assert causal.reader.use_matching is original_matching
