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


def test_training_empty_context_does_not_change_output_or_statistics():
    import copy
    head=RegionalRefinement().train()
    with torch.no_grad():head.head.conv_out4.conv.weight.fill_(.01)
    other=copy.deepcopy(head)
    x=list(inputs());changed=list(x)
    changed[1]=x[1].clone();changed[1][:,:,1]=1000
    assert torch.equal(head(*x),other(*changed))
    for key,value in head.state_dict().items():
        assert torch.equal(value,other.state_dict()[key]),key


def test_valid_projection_reuses_author_and_singleton_is_finite():
    import copy
    from modeling.regional_refinement import project_valid_context
    module=RegionalRefinement().head.conv_key.train()
    context=torch.randn(2,64,3,1)
    other=copy.deepcopy(module)
    assert torch.equal(module(context),project_valid_context(other,context,torch.ones(2,3,dtype=torch.bool)))
    before={k:v.clone() for k,v in module.state_dict().items()}
    single=project_valid_context(module,context[:1],torch.tensor([[True,False,False]]))
    assert torch.isfinite(single).all()
    assert single[:,:,1:].count_nonzero()==0
    assert module.training and all(m.training for m in module.modules())
    for key,value in before.items():assert torch.equal(value,module.state_dict()[key])
    single.sum().backward()
    assert any(p.grad is not None for p in module.parameters())


def test_verified_tensor_io_rejects_corruption_and_preserves_previous_file(tmp_path,monkeypatch):
    import hashlib
    from scripts import run_regional_refinement as runner
    path=tmp_path/'sample.pt'
    runner.save_tensor({'x':torch.arange(3)},path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert torch.equal(runner.load(path,sha)['x'],torch.arange(3))
    with pytest.raises(RuntimeError,match='SHA256 mismatch'):runner.load(path,'bad')
    original=path.read_bytes()
    def fail(*args,**kwargs):raise RuntimeError('interrupted write')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(RuntimeError):runner.save_tensor({},path)
    assert path.read_bytes()==original
    assert not list(tmp_path.glob('*.tmp'))
    path.write_bytes(b'PKbad')
    with pytest.raises(RuntimeError,match='sample.pt'):runner.load(path)
