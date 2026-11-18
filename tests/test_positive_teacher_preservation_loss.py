import pytest
import torch

from modeling.positive_teacher_preservation_loss import positive_teacher_penalty


def test_equal_prediction_and_empty_targets_have_zero_penalty():
    original=torch.tensor([[[[5.,-1024.],[1.,-2.]]]])
    target=torch.tensor([[[[1.,1.],[0.,0.]]]])
    assert positive_teacher_penalty(original,original,target)==0
    changed=torch.zeros_like(original,requires_grad=True)
    loss=positive_teacher_penalty(changed,original,torch.zeros_like(target));loss.backward()
    assert loss==0 and torch.equal(changed.grad,torch.zeros_like(changed))


def test_teacher_preservation_acts_only_on_positive_training_targets():
    original=torch.tensor([[[[5.,5.,-1024.]]]],requires_grad=True)
    prediction=torch.zeros_like(original,requires_grad=True)
    target=torch.tensor([[[[1.,0.,1.]]]],requires_grad=True)
    loss=positive_teacher_penalty(prediction,original,target);loss.backward()
    assert loss>0 and prediction.grad[0,0,0,0]<0
    assert prediction.grad[0,0,0,1]==0 and prediction.grad[0,0,0,2]==0
    assert original.grad is None and target.grad is None


def test_extreme_logits_are_finite_and_bad_inputs_rejected():
    for val in (-1024.,1024.):
        pred=torch.full((1,1,2,2),val,requires_grad=True)
        loss=positive_teacher_penalty(pred,torch.full_like(pred,-val),torch.ones_like(pred))
        loss.backward();assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    with pytest.raises(ValueError,match='align'):
        positive_teacher_penalty(torch.zeros(1),torch.zeros(1),torch.zeros(1))
    with pytest.raises(ValueError,match='finite'):
        positive_teacher_penalty(torch.full((1,1,1,1),float('nan')),torch.zeros(1,1,1,1),torch.zeros(1,1,1,1))


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS required')
def test_cpu_mps_values_and_gradients_match():
    original=torch.tensor([[[[5.,-1024.],[-2.,8.]]]])
    target=torch.tensor([[[[1.,1.],[0.,1.]]]])
    start=torch.tensor([[[[-1.,-1024.],[2.,4.]]]])
    results=[]
    for device in ('cpu','mps'):
        pred=start.to(device).detach().clone().requires_grad_();loss=positive_teacher_penalty(pred,original.to(device),target.to(device))
        loss.backward();results.append((loss.detach().cpu(),pred.grad.detach().cpu()))
    torch.testing.assert_close(results[0][0],results[1][0],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(results[0][1],results[1][1],rtol=1e-5,atol=1e-6)


def test_moderately_suppressed_teacher_is_not_promoted_to_full_confidence():
    target=torch.ones(1,1,10,10)
    prediction=torch.zeros_like(target)
    weak=positive_teacher_penalty(prediction,torch.full_like(target,-3.),target)
    strong=positive_teacher_penalty(prediction,torch.full_like(target,3.),target)
    assert weak < strong * 0.06
