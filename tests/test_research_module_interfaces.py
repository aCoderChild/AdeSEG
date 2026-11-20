"""Public scientific names must not fork trainable weights or alter ABI."""
import pytest
import torch

from adeseg.modules import FGPI, FGPIConfig, RMFA, RMFAConfig, TTMR, TTMRConfig
from modeling.structured_feature_aggregation import StructuredMultistageFeatureAggregator


def test_fgpi_constructs_joint_prompts_from_yolo_box_and_features():
    features = torch.ones(1, 4, 16, 16)
    prompts = FGPI().construct(features, [4, 4, 12, 12], (16, 16))
    assert prompts["mode"] == "fgpi"
    assert prompts["box_xyxy"] == [4., 4., 12., 12.]
    assert prompts["point_labels"] == [1, 0]
    assert prompts["anchor_count"] <= 4


def test_rmfa_reuses_source_bound_trainable_class():
    assert RMFA is StructuredMultistageFeatureAggregator
    model = RMFA(mode="foreground_background", early_channels=2,
                 deep_channels=4, hidden=8, attention_size=4)
    early = torch.randn(1, 2, 6, 6)
    deep = torch.randn(1, 4, 3, 3)
    logits = torch.randn(1, 1, 8, 8)
    assert torch.equal(model(early, deep, logits), logits)


def test_ttmr_is_a_fixed_budget_candidate_selector():
    torch.manual_seed(10)
    selector = TTMR(TTMRConfig(budget=3))
    features = torch.randn(6, 4)
    anchors = torch.randn(2, 4)
    usage = torch.tensor([.1, .2, .3, .4, .5, .6])
    frames = torch.tensor([0, 0, 1, 1, 2, 2])
    result = selector.select(features, anchors, usage, frames, current_frame=3)
    assert len(result["indices"]) == 3
    assert result["selected_token_count"] == 3
    assert result["status"] == "offline_shadow_selection_only"


@pytest.mark.parametrize("config", [
    FGPIConfig(positive_points=1, negative_points=1),
    RMFAConfig(mode="foreground_background"),
    TTMRConfig(budget=3),
])
def test_scientific_module_configuration_is_immutable(config):
    with pytest.raises(Exception):
        config.budget = 20
