import torch
import pytest
from modeling.regional_refinement import RegionalRefinement, pooled_context, refinement_loss


def inputs(batch=2):
    torch.manual_seed(7)
    return (torch.randn(batch,256,8,8),torch.randn(batch,64,3,1),
            torch.tensor([[True,False,True]]).expand(batch,-1),torch.randn(batch,1,32,32),torch.ones(batch))


def test_zero_residual_exact_and_absent_gate():
    head=RegionalRefinement().eval();x=inputs()
    assert torch.equal(head(*x),x[3])
    with torch.no_grad():head.head.conv_out4.conv.weight.fill_(.02)
    assert torch.equal(head(*(*x[:4],torch.zeros(2))),x[3])
    assert not torch.equal(head(*x),x[3])
    assert torch.isfinite(head(*x)).all()


def test_pooling_uses_spatial_rows_and_handles_empty_regions():
    features=torch.arange(4*64).float().reshape(4,64);labels=torch.tensor([0,0,2,2])
    context,valid,count=pooled_context(features,labels)
    assert torch.equal(context[:,0,0],features[:2].mean(0))
    assert torch.equal(context[:,2,0],features[2:].mean(0))
    assert context[:,1,0].count_nonzero()==0
    assert valid.tolist()==[True,False,True]
    assert count.tolist()==[2,0,2]
    with pytest.raises(ValueError):pooled_context(features,labels[:2])


def test_masked_empty_context_cannot_change_output_and_gradients_flow():
    head=RegionalRefinement().eval();x=list(inputs());target=torch.rand_like(x[3]).round()
    with torch.no_grad():head.head.conv_out4.conv.weight.fill_(.01)
    first=head(*x);x[1]=x[1].clone();x[1][:,:,1]=1000
    assert torch.equal(first,head(*x))
    loss=refinement_loss(first,target);loss.backward()
    assert torch.isfinite(loss)
    for module in (head.project,head.head.conv_query,head.head.conv_key,head.head.conv_value):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())


def test_all_empty_context_rejected():
    head=RegionalRefinement();x=list(inputs());x[2]=torch.zeros_like(x[2])
    with pytest.raises(ValueError):head(*x)
