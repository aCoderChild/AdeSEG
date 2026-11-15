import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from modeling.multiscale_memory_utility import attention_tile_mass, evaluate_scores, multiscale_scores


def test_multiscale_uses_spatial_context_and_aligned_tiles():
    torch.manual_seed(32)
    q = torch.randn(64, 6)
    k = q.clone()
    k.reshape(8, 8, 6)[:2, :2] *= -1
    result = multiscale_scores(q, k)
    assert all(x.shape == (16,) for x in result.values())
    assert result['scale_1'][0] < result['scale_1'][1]
    assert not np.array_equal(result['scale_1'], result['scale_4'])
    assert result['multiscale'][0] < result['multiscale'][-1]
    with pytest.raises(ValueError):
        multiscale_scores(q, k[:32])


def test_native_attention_matches_dense_softmax_including_pointers():
    torch.manual_seed(3)
    q = torch.randn(1, 2, 64, 8)
    k = torch.randn(1, 2, 64 * 2 + 3, 8)
    actual = attention_tile_mass(q, k, 64, 128, chunk_size=7)
    dense = (q @ k.transpose(-1, -2) / math.sqrt(8)).softmax(-1)
    mass = dense[..., 64:128].double().mean((0, 1, 2)).reshape(8, 8)
    expected = mass.reshape(4, 2, 4, 2).sum((1, 3)).reshape(-1).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
    without_pointers = attention_tile_mass(q, k[..., :-3, :], 64, 128)
    assert np.all(actual < without_pointers)


def test_utility_ties_and_sign_are_explicit():
    utility = np.arange(16, dtype=float)
    result = evaluate_scores(np.ones(16), utility)
    assert result['spearman_rho'] is None
    assert len(result['selected_tile_ids']) == 16
    assert result['selected_minus_uniform_dice'] == pytest.approx(0)
    result = evaluate_scores(utility, utility)
    assert result['spearman_rho'] == pytest.approx(1)
    assert result['selected_minus_uniform_dice'] == pytest.approx(7.5)
    with pytest.raises(ValueError):
        evaluate_scores(np.zeros(15), utility)


def test_summary_pairs_only_same_targets_when_correlations_missing():
    from scripts.run_multiscale_memory_utility_cohort import summarize

    def metrics(rho):
        return {'spearman_rho': rho, 'selected_minus_uniform_dice': 0.0, 'selected_tile_ids': [0]}
    rows = [
        {'sequence': 'seqA', 'pilot_pair': False, 'metrics': {'scale_1': metrics(0.1), 'multiscale': metrics(None), 'native_attention': metrics(0.0)}},
        {'sequence': 'seqA', 'pilot_pair': False, 'metrics': {'scale_1': metrics(0.2), 'multiscale': metrics(0.5), 'native_attention': metrics(0.0)}},
    ]
    summary = summarize(rows)['all']
    assert summary['multiscale_minus_local_spearman']['mean'] == pytest.approx(0.3)
