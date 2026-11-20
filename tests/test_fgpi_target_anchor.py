"""CPU-only checks of Lean-inspired box-derived pseudo-target anchors.

These tests validate geometry, causality-compatible inputs, fixed budgets, and
deterministic behavior. They do NOT establish segmentation accuracy.
"""
import random

import pytest
import torch

from adeseg.prompts.target_anchored import target_anchored_points


def synthetic_target():
    features=torch.zeros(1,3,24,24)
    features[:,1]=4
    features[:,0,8:16,8:16]=4
    features[:,1,8:16,8:16]=0
    features[:,1,6:8,6:18]=0
    features[:,2,6:8,6:18]=4  # distractor appearance at edge of box
    return features


def test_pseudo_anchor_points_follow_target_and_background_geometry():
    result=target_anchored_points(
        synthetic_target(),[6,6,18,18],(24,24),positives=2,negatives=2)
    assert result["point_labels"] == [1,1,0,0]
    assert result["anchor_count"] == len(result["anchor_coords"]) == 4
    assert len({tuple(x) for x in result["point_coords"]}) == 4
    assert all(8 <= x <= 16 and 8 <= y <= 16
               for x,y in result["point_coords"][:2])
    assert all(x<6 or x>18 or y<6 or y>18
               for x,y in result["point_coords"][2:])


def test_uniform_features_do_not_trigger_box_only_fallback():
    result=target_anchored_points(torch.ones(2,16,16),[4,4,12,12],(16,16))
    assert result["point_labels"] == [1,0]
    assert len(result["point_coords"]) == 2


def test_results_deterministic_and_invariant_to_positive_feature_scale():
    features=synthetic_target()
    original=target_anchored_points(features,[6,6,18,18],(24,24))
    scaled=target_anchored_points(features*5,[6,6,18,18],(24,24))
    assert original["point_coords"] == scaled["point_coords"]
    assert original == target_anchored_points(features,[6,6,18,18],(24,24))


def test_edge_aligned_box_still_selects_external_negative():
    result=target_anchored_points(torch.ones(2,16,16),[0,1,8,8],(16,16))
    assert result["point_labels"] == [1,0]


def test_full_frame_detector_box_has_no_trustworthy_negative_location():
    with pytest.raises(ValueError,match="Insufficient"):
        target_anchored_points(torch.ones(2,16,16),[0,0,16,16],(16,16))


def test_malformed_features_or_point_budget_raise():
    with pytest.raises(ValueError,match="finite"):
        target_anchored_points(torch.full((2,16,16),float("nan")),[4,4,12,12],(16,16))
    with pytest.raises(ValueError,match="budget"):
        target_anchored_points(torch.ones(2,16,16),[4,4,12,12],(16,16),positives=9)


def test_randomized_box_geometry_property():
    generator=random.Random(42)
    features=torch.randn(1,32,32,32,generator=torch.Generator().manual_seed(42))
    for _ in range(100):
        x0,y0=generator.randint(0,88),generator.randint(0,88)
        x1,y1=(x0+generator.randint(6,128-x0),
                y0+generator.randint(6,128-y0))
        out=target_anchored_points(features,[x0,y0,x1,y1],(128,128))
        assert out["point_labels"] == [1,0]
        (xp,yp),(xn,yn)=out["point_coords"]
        assert x0<=xp<=x1 and y0<=yp<=y1
        assert not(x0<=xn<=x1 and y0<=yn<=y1)
