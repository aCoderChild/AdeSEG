"""TTMR causal budgets, temporal coverage, novelty constraints and determinism."""
import pytest
import torch
from adeseg.memory.target_aware_temporal import TTMRConfig, rank_temporal_memory


def sample():
    torch.manual_seed(20261120)
    features=torch.randn(24,4)
    anchors=torch.randn(2,4)
    usage=torch.linspace(0,1,24)
    frames=torch.tensor([1]*8+[2]*8+[3]*8)
    return features,anchors,usage,frames


def test_fixed_budget_causal_temporal_spread_and_stable_order():
    f,a,u,t=sample()
    cfg=TTMRConfig(budget=6,minimum_per_frame=1)
    first=rank_temporal_memory(f,a,u,t,current_frame=4,config=cfg)
    second=rank_temporal_memory(f,a,u,t,current_frame=4,config=cfg)
    assert torch.equal(first["indices"],second["indices"])
    assert len(first["indices"].unique())==6
    assert set(first["retained_per_frame"])=={"1","2","3"}
    assert first["status"]=="offline_shadow_selection_only"


def test_budget_smaller_than_frame_groups_preserves_temporal_span():
    f,a,u,t=sample()
    chosen=rank_temporal_memory(f,a,u,t,current_frame=4,config=TTMRConfig(budget=2))
    assert set(chosen["retained_per_frame"])=={"1","3"}


def test_no_future_frame_or_invalid_budget_is_allowed():
    f,a,u,t=sample()
    with pytest.raises(ValueError,match="historical"):
        rank_temporal_memory(f,a,u,t,current_frame=3,config=TTMRConfig(budget=4))
    with pytest.raises(ValueError,match="budget"):
        TTMRConfig(budget=0)


def test_maximum_frame_cap_enforced_and_infeasible_cap_fails():
    f,a,u,t=sample()
    result=rank_temporal_memory(f,a,u,t,current_frame=4,
                                 config=TTMRConfig(budget=6,maximum_per_frame=2))
    assert all(count<=2 for count in result["retained_per_frame"].values())
    with pytest.raises(ValueError,match="cap"):
        rank_temporal_memory(f,a,u,t,current_frame=4,
                             config=TTMRConfig(budget=7,maximum_per_frame=2))


def test_multiscale_scores_and_nan_inputs_fail_closed():
    f,a,u,t=sample()
    bad=torch.ones(24)*float("nan")
    with pytest.raises(ValueError,match="finite"):
        rank_temporal_memory(f,a,u,t,current_frame=4,config=TTMRConfig(budget=4),
                             multiscale_scores=bad)
    with pytest.raises(ValueError,match="pre-normalized"):
        rank_temporal_memory(f,a,u*100,t,current_frame=4,config=TTMRConfig(budget=4))
