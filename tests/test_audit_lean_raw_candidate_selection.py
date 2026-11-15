import pytest
import torch

from scripts.audit_lean_raw_candidate_selection import feature_scores


def test_feature_scores_prefer_foreground_and_support_background_contrast():
    # First position is FG-like, second BG-like; candidates have opposite masks.
    feature=torch.tensor([[[1.,0.]],[[0.,1.]]])
    foreground=torch.tensor([1.,0.]);background=torch.tensor([0.,1.])
    masks=torch.tensor([[[[10.,-10.]],[[-10.,10.]]]])
    fg,contrast=feature_scores(masks,feature,foreground,background)
    assert fg[0]>fg[1] and contrast[0]>contrast[1]


def test_feature_scores_reject_wrong_geometry_and_nonfinite():
    feature=torch.randn(2,1,2);fg=torch.randn(2);bg=torch.randn(2)
    with pytest.raises(ValueError,match='geometry'):
        feature_scores(torch.zeros(1,3,2,2),feature,fg,bg)
    feature[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        feature_scores(torch.zeros(1,3,1,2),feature,fg,bg)
