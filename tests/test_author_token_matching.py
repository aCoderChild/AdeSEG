import pytest
import torch
import torch.nn.functional as F

from modeling.author_token_matching import matching_backend, use_matching
from modeling.region_constrained_tome import compress_memory_kv


@pytest.mark.parametrize('strategy', ['tome', 'atc'])
def test_weighted_merge_conserves_mass_constant_values_and_distinct_value_width(strategy):
    torch.manual_seed(12)
    metric = torch.randn(1, 16, 7)
    backend = matching_backend(strategy)
    merge, _ = backend.bipartite_soft_matching(metric, 8)
    values, mass = backend.merge_wavg(merge, torch.full((1, 16, 3), 5.))
    source = backend.merge_source(merge, metric)
    assert values.shape == (1, 8, 3)
    assert torch.equal(values, torch.full_like(values, 5.))
    assert mass.sum() == 16
    assert torch.equal(source.sum(1), torch.ones(1, 16))
    assert torch.equal(source.sum(-1), mass[..., 0])
    reconstructed = source.transpose(1, 2) @ values
    assert torch.equal(reconstructed, torch.full((1, 16, 3), 5.))


def test_atc_and_tome_match_budgets_keep_pointers_and_preserve_shared_key_value_assignments():
    torch.manual_seed(13)
    keys = torch.randn(1, 2, 18, 8)
    values = 2 * keys
    roles = F.one_hot(torch.tensor([[0, 1, 2, 0] * 4]), 3).float()
    spans = [{'frame': 3, 't_pos': 0, 'start': 0, 'stop': 16}]
    results = []
    for strategy in ('tome', 'atc'):
        with use_matching(strategy):
            for mode in ('regional', 'global'):
                k, v, bias, audit = compress_memory_kv(keys, values, roles, spans,
                    mode=mode, pointers=2, window_size=2)
                assert torch.equal(k[..., -2:, :], keys[..., -2:, :])
                assert torch.equal(v[..., -2:, :], values[..., -2:, :])
                torch.testing.assert_close(v, 2 * k)
                torch.testing.assert_close(bias.exp().sum(), torch.tensor(18.))
                results.append((k.shape, audit['frames'][0]['windows']))
    assert all(result == results[0] for result in results)


def test_atc_adapter_rejects_unsupported_counts_and_zero_metrics():
    backend = matching_backend('atc')
    with pytest.raises(ValueError):
        backend.bipartite_soft_matching(torch.ones(1, 8, 3), 5)
    with pytest.raises(ValueError):
        backend.bipartite_soft_matching(torch.zeros(1, 8, 3), 4)
