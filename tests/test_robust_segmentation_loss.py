import pytest
import torch
import torch.nn.functional as F

from modeling.robust_segmentation_loss import robust_bce_dice_loss


def test_extreme_false_negative_is_finite_and_has_smaller_nonzero_gradient():
    logits=torch.full((1,1,2,2),-1024.,requires_grad=True);target=torch.ones_like(logits)
    loss=robust_bce_dice_loss(logits,target);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    # Dice gradient saturates here. Each pixel receives mean-reduced robust BCE
    # gradient -10/(10+1024)/4 rather than the standard BCE gradient -1/4.
    torch.testing.assert_close(logits.grad,torch.full_like(logits,-10/(10+1024)/4),atol=1e-7,rtol=1e-6)


def test_small_error_remains_close_to_bce_plus_dice():
    logits=torch.full((1,1,2,2),7.);target=torch.ones_like(logits)
    prob=logits.sigmoid();ordinary=F.binary_cross_entropy_with_logits(logits,target)+1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
    torch.testing.assert_close(robust_bce_dice_loss(logits,target),ordinary,atol=1e-6,rtol=1e-4)


def test_invalid_target_and_geometry_rejected():
    pred=torch.zeros(1,1,2,2)
    with pytest.raises(ValueError):robust_bce_dice_loss(pred,torch.ones(1,1,3,3))
    with pytest.raises(ValueError):robust_bce_dice_loss(pred,torch.full_like(pred,2.))
    with pytest.raises(ValueError):robust_bce_dice_loss(pred,torch.zeros_like(pred),scale=float('nan'))


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS required')
def test_mps_loss_and_gradients_agree_for_extreme_inputs():
    source=torch.tensor([[[[-1024.,-30.],[0.,7.]]]])
    target=torch.tensor([[[[1.,0.],[1.,1.]]]])
    values=[]
    for device in ('cpu','mps'):
        x=source.detach().to(device).requires_grad_(True);loss=robust_bce_dice_loss(x,target.to(device));loss.backward()
        values.append((loss.detach().cpu(),x.grad.detach().cpu()))
    torch.testing.assert_close(values[0][0],values[1][0],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(values[0][1],values[1][1],atol=1e-6,rtol=1e-5)
