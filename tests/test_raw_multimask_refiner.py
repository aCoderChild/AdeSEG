import hashlib
from pathlib import Path

import pytest
import torch

import modeling.foreground_conditioned_refiner as conditioned
import modeling.multilevel_contrast_refiner as base
import modeling.raw_multimask_refiner as raw_module
from modeling.lean_channel_attention import author_sha256
from modeling.raw_multimask_refiner import RawMultimaskRefiner


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    return (torch.randn(1,32,8,8),torch.randn(1,256,2,2),torch.randn(1,1,8,8),
            torch.randn(1,3,8,8),torch.randn(1,3))


def payload(model):
    return {'format':'raw_multimask_refiner_v1','research_only':True,'training_steps':1,'state_dict':model.state_dict(),
            'raw_model_sha256':sha(raw_module.__file__),'conditioned_source_sha256':sha(conditioned.__file__),
            'base_source_sha256':sha(base.__file__),'author_sha256':author_sha256(),
            'backbone_sha256':'a'*64,'feature_pipeline_sha256':'b'*64,'train_manifest_sha256':'c'*64,'raw_manifest_sha256':'d'*64,
            'raw_mode':model.raw_mode,'gate_mode':model.gate_mode,'contrast_head':model.contrast_head}


def load(model,data):
    model.load_trained_state(data,backbone_sha256='a'*64,feature_pipeline_sha256='b'*64,
                             train_manifest_sha256='c'*64,raw_manifest_sha256='d'*64)


@pytest.mark.parametrize('mode',['raw','zero','spatial_shift'])
def test_zero_initialization_preserves_native_logits(mode):
    model=RawMultimaskRefiner(raw_mode=mode).train();x=list(inputs())
    for value in (-1024.,0.,1024.):
        x[2]=torch.full_like(x[2],value)
        result=model(*x)
        assert torch.isfinite(result).all() and torch.equal(result,x[2])


def test_raw_controls_are_capacity_matched_and_zero_is_input_invariant():
    models=[RawMultimaskRefiner(raw_mode=mode) for mode in ('raw','zero','spatial_shift')]
    assert len({sum(p.numel() for p in m.parameters()) for m in models})==1
    model=models[1].train();torch.nn.init.normal_(model.raw_adapter.weight,std=.1);torch.nn.init.normal_(model.head[-1].weight,std=.01)
    x=list(inputs());first=model(*x)
    x[3]=torch.randn_like(x[3])*100;x[4]=torch.randn_like(x[4])*100
    assert torch.equal(first,model(*x))


def test_shift_changes_only_candidate_map_alignment():
    raw=RawMultimaskRefiner(raw_mode='raw');shift=RawMultimaskRefiner(raw_mode='spatial_shift')
    masks=torch.randn(1,3,8,8);ious=torch.randn(1,3)
    left,right=raw.raw_input(masks,ious),shift.raw_input(masks,ious)
    assert torch.equal(right[:,:3],left[:,:3].roll(97,dims=-1))
    assert torch.equal(right[:,3:],left[:,3:])


def test_raw_adapter_receives_gradient_and_geometry_is_checked():
    model=RawMultimaskRefiner().train();torch.nn.init.normal_(model.head[-1].weight,std=.01)
    x=list(inputs());model(*x).square().mean().backward()
    assert model.raw_adapter.weight.grad is not None and model.raw_adapter.weight.grad.abs().sum()>0
    x[3]=torch.randn(1,2,8,8)
    with pytest.raises(ValueError,match='geometry'):model(*x)


def test_distinct_checkpoint_format_and_failure_revokes_inference():
    model=RawMultimaskRefiner().eval();x=inputs()
    with pytest.raises(RuntimeError,match='Untrained'):
        with torch.no_grad():model(*x)
    old=payload(model);old['format']='foreground_conditioned_refiner_v1'
    with pytest.raises(ValueError,match='checkpoint'):load(model,old)
    good=payload(model);load(model,good)
    with torch.no_grad():assert torch.equal(model(*x),x[2])
    bad=payload(model);bad['raw_mode']='zero'
    with pytest.raises(ValueError,match='ablation'):load(model,bad)
    with pytest.raises(RuntimeError,match='Untrained'):
        with torch.no_grad():model(*x)


def test_existing_three_input_live_route_is_explicitly_rejected():
    model=RawMultimaskRefiner().train();early,deep,logits,_,_=inputs()
    with pytest.raises(RuntimeError,match='existing live router does not support'):
        model(early,deep,logits)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS required')
def test_cpu_mps_raw_forward_and_gradients_agree():
    torch.manual_seed(32);cpu=RawMultimaskRefiner().train()
    torch.nn.init.normal_(cpu.head[-1].weight,std=.01)
    torch.nn.init.normal_(cpu.raw_adapter.weight,std=.01)
    mps=RawMultimaskRefiner().train();mps.load_state_dict(cpu.state_dict());mps.to('mps')
    x=inputs();values=[]
    for model,device in ((cpu,'cpu'),(mps,'mps')):
        out=model(*(t.to(device) for t in x));loss=out.square().mean();loss.backward()
        values.append((out.detach().cpu(),[p.grad.detach().cpu() for p in model.parameters()]))
    torch.testing.assert_close(values[0][0],values[1][0],atol=1e-4,rtol=1e-4)
    for left,right in zip(values[0][1],values[1][1],strict=True):
        torch.testing.assert_close(left,right,atol=1e-4,rtol=1e-4)
