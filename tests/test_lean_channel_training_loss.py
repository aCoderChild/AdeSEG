import pytest
import torch
from modeling.lean_channel_training_loss import balanced_soft_bce


def test_balanced_loss_rewards_correct_logits_for_positive_and_empty_frames():
    target=torch.tensor([1.,1.,0.,0.]);correct=torch.tensor([.5,.4,-.5,-.4],requires_grad=True)
    wrong=-correct.detach();loss=balanced_soft_bce(correct,target)
    assert loss<balanced_soft_bce(wrong,target);loss.backward();assert correct.grad is not None
    empty=torch.zeros(4);assert balanced_soft_bce(torch.full((4,),-.5),empty)<balanced_soft_bce(torch.full((4,),.5),empty)


def test_soft_occupancy_and_invalid_targets():
    assert torch.isfinite(balanced_soft_bce(torch.tensor([0.,.2]),torch.tensor([.25,.75])))
    with pytest.raises(ValueError):balanced_soft_bce(torch.zeros(2),torch.tensor([0.,2.]))
