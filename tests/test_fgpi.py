"""Pure CPU FGPI regression: detector-only versus feature-augmented prompts."""
import pytest
import torch
from adeseg.prompts.feature_guided import feature_guided_prompt, apply_fgpi_prompt


def feature_grid():
    features = torch.zeros(1, 2, 16, 16)
    features[:, 1] = 4
    features[:, 0, 4:12, 4:12] = 4
    features[:, 1, 4:12, 4:12] = 0
    return features


def test_detector_only_preserves_geometry_without_features():
    p = feature_guided_prompt(None, [4,4,12,12], (16,16), mode="box_only")
    assert p["box_xyxy"] == [4., 4., 12., 12.]
    assert p["point_coords"] == []
    assert p["decision"] == "box_only"


def test_feature_prompts_are_inside_box_positive_and_outside_negative():
    p = feature_guided_prompt(feature_grid(), [4,4,12,12], (16,16))
    assert p["decision"] == "augmented"
    assert p["point_labels"] == [1,0]
    x,y=p["point_coords"][0]
    assert 4 <= x <= 12 and 4 <= y <= 12
    x,y=p["point_coords"][1]
    assert x < 4 or x > 12 or y < 4 or y > 12
    assert p["feature_contrast"] > 0


def test_uniform_features_fall_back_without_making_up_points():
    p = feature_guided_prompt(torch.ones(2, 16,16),[4,4,12,12],(16,16))
    assert p["decision"] == "box_only"
    assert p["point_coords"] == []
    assert p["reason"] == "weak_feature_discrimination"


def test_nonfinite_features_and_invalid_detector_box_rejected():
    with pytest.raises(ValueError,match="finite float"):
        feature_guided_prompt(torch.full((2,16,16),float("nan")),[4,4,12,12],(16,16))
    with pytest.raises(ValueError,match="outside"):
        feature_guided_prompt(feature_grid(),[-1,4,12,12],(16,16))
    with pytest.raises(ValueError,match="confidence"):
        feature_guided_prompt(feature_grid(),[4,4,12,12],(16,16),
                              detector_confidence=0.1)


def test_sam2_bridge_sends_detector_box_first_and_points_without_gt():
    class FakePredictor:
        def add_new_points_or_box(self, state, **kwargs):
            assert state == "session"
            assert kwargs["clear_old_points"] and kwargs["normalize_coords"]
            assert kwargs["box"] == [4.,4.,12.,12.]
            assert kwargs["labels"] == [1,0]
            return kwargs
    p=feature_guided_prompt(feature_grid(),[4,4,12,12],(16,16))
    out=apply_fgpi_prompt(FakePredictor(),"session",0,1,p)
    assert out["frame_idx"] == 0
