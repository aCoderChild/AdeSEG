"""Regression tests for multiscale scoring without extra memory tokens."""
import pytest
import torch
import torch.nn.functional as F

from adeseg.modules.ttmr import TTMR, TTMRConfig, rank_spatial_memory
from adeseg.modules.ttmr.multiscale import score_multiscale_memory


def _sample():
    torch.manual_seed(20261120)
    maps = torch.randn(2, 4, 4, 4)
    anchors = torch.randn(3, 4)
    usage = torch.linspace(0.05, 0.95, 32)
    frames = torch.tensor([1, 3], dtype=torch.long)
    return maps, anchors, usage, frames


def test_zero_context_weight_matches_flat_original_token_similarity():
    maps, anchors, _, _ = _sample()
    flattened = maps.permute(0, 2, 3, 1).reshape(-1, 4)
    expected = (
        F.normalize(flattened, dim=-1)
        @ F.normalize(anchors, dim=-1).T
    ).amax(dim=-1)
    result = score_multiscale_memory(maps, anchors, context_weight=0)
    assert result.shape == (32,)
    assert torch.allclose(result, expected, atol=1e-6)


def test_multiscale_scores_change_without_changing_original_memory():
    maps, anchors, _, _ = _sample()
    before = maps.clone()
    local = score_multiscale_memory(maps, anchors, context_weight=0)
    mixed = score_multiscale_memory(maps, anchors, context_weight=0.5)
    assert mixed.shape == local.shape
    assert torch.isfinite(mixed).all()
    assert (mixed.abs() <= 1 + 1e-6).all()
    assert not torch.allclose(local, mixed)
    assert torch.equal(maps, before)


def test_public_spatial_selector_preserves_fixed_budget_and_causality():
    maps, anchors, usage, frames = _sample()
    selector = TTMR(TTMRConfig(budget=6, minimum_per_frame=1))
    first = selector.select_spatial(
        maps, anchors, usage, frames, current_frame=4
    )
    second = selector.select_spatial(
        maps, anchors, usage, frames, current_frame=4
    )
    assert torch.equal(first["indices"], second["indices"])
    assert first["selected_token_count"] == 6
    assert first["candidate_token_count"] == 32
    assert first["indices"].unique().numel() == 6
    assert set(first["retained_per_frame"]) == {"1", "3"}
    assert first["status"] == "offline_shadow_selection_only"


def test_zero_multiscale_selection_weight_is_exact_native_anchor_reference():
    maps, anchors, usage, frames = _sample()
    config = TTMRConfig(budget=6, multiscale_weight=0)
    selector = TTMR(config)
    spatial = selector.select_spatial(maps, anchors, usage, frames, current_frame=4)
    flat = maps.permute(0, 2, 3, 1).reshape(-1, 4)
    baseline = selector.select(
        flat, anchors, usage, frames.repeat_interleave(16), current_frame=4
    )
    assert torch.equal(spatial["indices"], baseline["indices"])
    assert torch.equal(spatial["selected_scores"], baseline["selected_scores"])


def test_score_and_frame_input_rejections_fail_closed():
    maps, anchors, usage, frames = _sample()
    with pytest.raises(ValueError, match="scales"):
        score_multiscale_memory(maps, anchors, scales=(3,))
    with pytest.raises(ValueError, match="context_weight"):
        score_multiscale_memory(maps, anchors, context_weight=1.5)
    with pytest.raises(ValueError, match="anchors"):
        score_multiscale_memory(maps, anchors[:, :2])
    with pytest.raises(ValueError, match="frame IDs"):
        rank_spatial_memory(
            maps, anchors, usage, torch.tensor([1, 2, 3]),
            current_frame=4, config=TTMRConfig(budget=6)
        )
    with pytest.raises(ValueError, match="historical"):
        rank_spatial_memory(
            maps, anchors, usage, torch.tensor([1, 4]),
            current_frame=4, config=TTMRConfig(budget=6)
        )
    with pytest.raises(ValueError, match="multiscale_weight"):
        TTMRConfig(budget=6, multiscale_weight=1.5)
