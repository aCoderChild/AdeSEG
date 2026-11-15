import numpy as np
import pytest
import torch
from modeling.constrained_prompt_feature_refinement import constrained_early_feature_points


def test_points_follow_automatic_mask_and_box_on_nonsquare_image():
    torch.manual_seed(42)
    feature = torch.rand((1, 3, 8, 8))
    mask = np.zeros((24, 40), bool)
    mask[9:15, 15:25] = True
    points, status = constrained_early_feature_points({'backbone_fpn': [feature]}, mask, [5, 3, 35, 21])
    assert status == 'available'
    pos, neg = points['feature_points']
    assert mask[int(pos[1]), int(pos[0])]
    assert not mask[int(neg[1]), int(neg[0])]
    assert all(5 <= x < 35 and 3 <= y < 21 for x, y in (pos, neg))


def test_no_outside_candidate_falls_back_without_changing_box():
    torch.manual_seed(1)
    mask = np.zeros((24, 40), bool); mask[6:18, 10:30] = True
    points, status = constrained_early_feature_points({'backbone_fpn': [torch.rand((1, 3, 24, 40))]}, mask, [12, 8, 28, 16])
    assert points is None
    assert status == 'insufficient_foreground_or_background_cells'


def test_invalid_and_empty_masks():
    feature = {'backbone_fpn': [torch.ones((1, 3, 8, 8))]}
    assert constrained_early_feature_points(feature, np.zeros((16, 16), bool), [0, 0, 16, 16])[1] == 'empty_provisional_mask'
    with pytest.raises(ValueError):
        constrained_early_feature_points(feature, np.ones((2, 2, 2)), [0, 0, 1, 1])
