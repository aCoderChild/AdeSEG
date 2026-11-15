import hashlib
from pathlib import Path

import pytest
import torch

from modeling.foreground_conditioned_refiner import ForegroundConditionedRefiner
from modeling.lean_channel_attention import author_sha256
import modeling.foreground_conditioned_refiner as candidate
import modeling.multilevel_contrast_refiner as base


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload(model, *, mode='region'):
    return {'format':'foreground_conditioned_refiner_v1','research_only':True,'training_steps':1,
            'author_sha256':author_sha256(),'conditioned_source_sha256':sha(candidate.__file__),
            'base_source_sha256':sha(base.__file__),'gate_mode':mode,'contrast_head':True,
            'backbone_sha256':'a'*64,'feature_pipeline_sha256':'b'*64,'train_manifest_sha256':'c'*64,
            'arm':'multilevel','state_dict':model.state_dict()}


def load(model, data):
    model.load_trained_state(data,backbone_sha256='a'*64,feature_pipeline_sha256='b'*64,train_manifest_sha256='c'*64)


@pytest.mark.parametrize('mode', ['region', 'global', 'shuffled'])
def test_zero_residual_preserves_logits_and_extremes(mode):
    model=ForegroundConditionedRefiner(gate_mode=mode).train()
    early=torch.randn(1,32,8,8);deep=torch.randn(1,256,2,2)
    for value in (-1024.,0.,1024.):
        logits=torch.full((1,1,8,8),value)
        result=model(early,deep,logits)
        assert torch.isfinite(result).all() and torch.equal(result,logits)


@pytest.mark.parametrize('mode', ['region', 'global', 'shuffled'])
def test_weights_exactly_match_author_cbam_gate(mode):
    model=ForegroundConditionedRefiner(gate_mode=mode)
    x=torch.randn(2,64,4,4);fg=torch.randn(2,64,1,1);bg=torch.randn(2,64,1,1)
    weights=model.channel_weights(x,fg,bg)
    gate_input=x if mode=='global' else torch.cat((fg,-bg),dim=-1)
    if mode=='shuffled':gate_input=gate_input.roll(1,dims=1)
    torch.testing.assert_close(model.gate(gate_input),gate_input*weights,rtol=0,atol=0)
    assert weights.shape==(2,64,1,1) and (weights>0).all() and (weights<1).all()


def test_modes_are_capacity_matched_and_region_descriptors_affect_weights():
    region=ForegroundConditionedRefiner(gate_mode='region')
    control=ForegroundConditionedRefiner(gate_mode='global')
    shuffled=ForegroundConditionedRefiner(gate_mode='shuffled')
    assert sum(p.numel() for p in region.parameters())==sum(p.numel() for p in control.parameters())==sum(p.numel() for p in shuffled.parameters())
    # Set MLP weights to a deterministic positive function of its inputs.
    with torch.no_grad():
        for layer in region.gate.mlp:
            if isinstance(layer,torch.nn.Linear):layer.weight.fill_(0.01);layer.bias.zero_()
    x=torch.ones(1,64,2,2);bg=torch.zeros(1,64,1,1)
    low=region.channel_weights(x,torch.zeros_like(bg),bg)
    high=region.channel_weights(x,torch.ones_like(bg),bg)
    assert (high>low).all()


def test_untrained_eval_rejected_and_prior_format_cannot_load():
    model=ForegroundConditionedRefiner().eval()
    early=torch.randn(1,32,8,8);deep=torch.randn(1,256,2,2);logits=torch.zeros(1,1,8,8)
    with pytest.raises(RuntimeError,match='Untrained'):
        with torch.no_grad():model(early,deep,logits)
    bad=payload(model);bad['format']='multilevel_contrast_refiner_v1'
    with pytest.raises(ValueError,match='checkpoint'):
        load(model,bad)
    good=payload(model);load(model,good)
    with torch.no_grad():assert torch.equal(model(early,deep,logits),logits)
    bad=payload(model);bad['gate_mode']='global'
    with pytest.raises(ValueError,match='ablation'):
        load(model,bad)
    with pytest.raises(RuntimeError,match='Untrained'):
        with torch.no_grad():model(early,deep,logits)


def test_conditional_gate_and_residual_head_have_finite_gradients():
    torch.manual_seed(5)
    model=ForegroundConditionedRefiner().train()
    # The base head's final layer is deliberately initialized to zero, so
    # change it here only to check gradients can reach the gate and features.
    torch.nn.init.normal_(model.head[-1].weight,mean=0,std=0.01)
    early=torch.randn(1,32,8,8,requires_grad=True);deep=torch.randn(1,256,2,2,requires_grad=True)
    logits=torch.randn(1,1,8,8);model(early,deep,logits).square().mean().backward()
    assert early.grad is not None and torch.isfinite(early.grad).all() and early.grad.abs().sum()>0
    assert deep.grad is not None and torch.isfinite(deep.grad).all() and deep.grad.abs().sum()>0
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.gate.parameters())


def test_suppressed_object_logits_have_finite_training_gradients():
    model=ForegroundConditionedRefiner().train()
    early=torch.randn(1,32,8,8);deep=torch.randn(1,256,2,2)
    # A real hard training read was suppressed to -1024 by SAM2's object
    # presence logic. Training must remain finite even though this test does
    # not establish that the refiner can recover that object in video.
    logits=torch.full((1,1,8,8),-1024.)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(early,deep,logits),torch.ones_like(logits))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head[-1].weight.grad.abs().sum()>0


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS required')
def test_cpu_mps_training_forward_and_gradient_agree():
    torch.manual_seed(17)
    cpu=ForegroundConditionedRefiner().train()
    torch.nn.init.normal_(cpu.head[-1].weight,mean=0,std=0.01)
    mps=ForegroundConditionedRefiner().train()
    mps.load_state_dict(cpu.state_dict())
    mps.to('mps')
    early=torch.randn(1,32,8,8);deep=torch.randn(1,256,2,2);logits=torch.randn(1,1,8,8)
    outputs=[]
    for model,device in ((cpu,'cpu'),(mps,'mps')):
        out=model(early.to(device),deep.to(device),logits.to(device))
        loss=out.square().mean();loss.backward()
        outputs.append((out.detach().cpu(),[p.grad.detach().cpu() for p in model.parameters()]))
    torch.testing.assert_close(outputs[0][0],outputs[1][0],atol=1e-4,rtol=1e-4)
    for left,right in zip(outputs[0][1],outputs[1][1],strict=True):
        torch.testing.assert_close(left,right,atol=1e-4,rtol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS required')
def test_cpu_mps_suppressed_logits_training_agrees():
    torch.manual_seed(19)
    cpu=ForegroundConditionedRefiner().train()
    mps=ForegroundConditionedRefiner().train()
    mps.load_state_dict(cpu.state_dict());mps.to('mps')
    early=torch.randn(1,32,8,8);deep=torch.randn(1,256,2,2)
    logits=torch.full((1,1,8,8),-1024.)
    results=[]
    for model,device in ((cpu,'cpu'),(mps,'mps')):
        out=model(early.to(device),deep.to(device),logits.to(device))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(out,torch.ones_like(out));loss.backward()
        grads=[None if p.grad is None else p.grad.detach().cpu() for p in model.parameters()]
        results.append((loss.detach().cpu(),grads))
    torch.testing.assert_close(results[0][0],results[1][0],atol=1e-3,rtol=1e-3)
    for left,right in zip(results[0][1],results[1][1],strict=True):
        assert (left is None)==(right is None)
        if left is not None:torch.testing.assert_close(left,right,atol=1e-3,rtol=1e-3)
