import numpy as np
import pytest
import torch

from modeling.early_feature_prompt import early_feature_points


def test_early_fpn_selects_both_box_points_without_labels():
    feature = torch.zeros(1, 2, 16, 16)
    feature[:, 0] = 1
    feature[:, 0, 4:8, 4:8] = 0
    feature[:, 1, 4:8, 4:8] = 1
    mask = np.zeros((160, 160), dtype=bool)
    mask[40:80, 40:80] = True
    out, reason = early_feature_points({'backbone_fpn': [feature]}, mask, [20., 20., 120., 120.], 42)
    assert reason == 'available'
    assert len(out['feature_points']) == 2
    positive, negative = out['feature_points']
    assert 40 <= positive[0] < 80 and 40 <= positive[1] < 80
    assert 20 <= negative[0] < 120 and 20 <= negative[1] < 120
    assert positive != negative


def test_empty_mask_and_nonfinite_features_are_distinct():
    out, reason = early_feature_points({'backbone_fpn': [torch.ones(1, 2, 16, 16)]},
                                       np.zeros((16, 16), bool), [0, 0, 16, 16], 0)
    assert out is None and reason == 'empty_provisional_mask'
    with pytest.raises(ValueError, match='Nonfinite'):
        early_feature_points({'backbone_fpn': [torch.full((1, 2, 16, 16), float('nan'))]},
                             np.zeros((16, 16), bool), [0, 0, 16, 16], 0)
