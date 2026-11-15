import torch
from scripts.audit_bounded_anchor_v1 import compare


def test_suppressed_anchor_keeps_mask_but_changes_confidence():
    row=compare(torch.tensor([[[[-1024., -1024.]]]]),(1,2))
    assert row['mask_changed_pixels']==0 and row['native_positive_pixels']==0
    assert row['bounded_logit_min']==-16.0
    assert row['mean_bounded_probability']>row['mean_native_probability']


def test_nonlinearity_before_interpolation_can_change_mask():
    row=compare(torch.tensor([[[[-100.,10.]]]]),(1,7))
    assert row['mask_changed_pixels']>0
    assert row['bounded_only_pixels']>0
