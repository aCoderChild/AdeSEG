import pytest
import torch
from modeling.multilevel_contrast_refiner import MultilevelContrastRefiner
from modeling.lean_channel_attention import author_sha256


def test_zero_residual_is_exact_and_trainable():
    model=MultilevelContrastRefiner();a=torch.randn(1,32,8,8);b=torch.randn(1,256,2,2);m=torch.randn(1,1,8,8)
    out=model(a,b,m);assert torch.equal(out,m)
    out.mean().backward();assert model.head[-1].weight.grad is not None


def test_empty_extreme_masks_finite_and_eval_rejects_untrained():
    model=MultilevelContrastRefiner();a=torch.randn(1,32,8,8);b=torch.randn(1,256,2,2)
    for value in (-100.,100.):assert torch.isfinite(model(a,b,torch.full((1,1,8,8),value))).all()
    model.eval()
    with pytest.raises(RuntimeError):model(a,b,torch.zeros(1,1,8,8))


def test_bad_geometry_and_checkpoint_fail_closed():
    model=MultilevelContrastRefiner()
    with pytest.raises(ValueError):model(torch.zeros(1,31,8,8),torch.zeros(1,256,2,2),torch.zeros(1,1,8,8))
    with pytest.raises(ValueError):model.load_trained_state({'training_steps':0},backbone_sha256='a'*64,feature_pipeline_sha256='b'*64,train_manifest_sha256='c'*64)


def test_checkpoint_requires_matching_feature_provenance():
    model=MultilevelContrastRefiner()
    payload={'format':'multilevel_contrast_refiner_v1','training_steps':1,'research_only':True,'state_dict':model.state_dict(),'backbone_sha256':'a'*64,'feature_pipeline_sha256':'b'*64,'train_manifest_sha256':'c'*64,'author_sha256':author_sha256()}
    with pytest.raises(ValueError):model.load_trained_state(payload,backbone_sha256='d'*64,feature_pipeline_sha256='b'*64,train_manifest_sha256='c'*64)
    model.load_trained_state(payload,backbone_sha256='a'*64,feature_pipeline_sha256='b'*64,train_manifest_sha256='c'*64)
    model.eval()
    assert model(torch.zeros(1,32,8,8),torch.zeros(1,256,2,2),torch.zeros(1,1,8,8)).shape==(1,1,8,8)


def test_untrained_no_grad_and_failed_reload_fail_closed():
    model=MultilevelContrastRefiner()
    inputs=(torch.zeros(1,32,8,8),torch.zeros(1,256,2,2),torch.zeros(1,1,8,8))
    with torch.no_grad(), pytest.raises(RuntimeError):model(*inputs)
    payload={'format':'multilevel_contrast_refiner_v1','training_steps':1,'research_only':True,'state_dict':model.state_dict(),'backbone_sha256':'a'*64,'feature_pipeline_sha256':'b'*64,'train_manifest_sha256':'c'*64,'author_sha256':author_sha256()}
    model.load_trained_state(payload,backbone_sha256='a'*64,feature_pipeline_sha256='b'*64,train_manifest_sha256='c'*64)
    model.eval()
    with pytest.raises(ValueError):model.load_trained_state(payload,backbone_sha256='d'*64,feature_pipeline_sha256='b'*64,train_manifest_sha256='c'*64)
    with pytest.raises(RuntimeError):model(*inputs)
    model.load_trained_state(payload,backbone_sha256='a'*64,feature_pipeline_sha256='b'*64,train_manifest_sha256='c'*64)
    broken=dict(payload);broken['state_dict']=dict(payload['state_dict']);broken['state_dict'].pop(next(iter(broken['state_dict'])))
    with pytest.raises(RuntimeError):model.load_trained_state(broken,backbone_sha256='a'*64,feature_pipeline_sha256='b'*64,train_manifest_sha256='c'*64)
    with pytest.raises(RuntimeError):model(*inputs)


def test_mask_only_checkpoint_enforces_ablation_during_every_inference():
    model=MultilevelContrastRefiner()
    torch.nn.init.normal_(model.head[-1].weight)
    payload={'format':'multilevel_contrast_refiner_v1','training_steps':1,'research_only':True,'state_dict':model.state_dict(),'backbone_sha256':'a'*64,'feature_pipeline_sha256':'b'*64,'train_manifest_sha256':'c'*64,'author_sha256':author_sha256(),'arm':'mask_only'}
    model.load_trained_state(payload,backbone_sha256='a'*64,feature_pipeline_sha256='b'*64,train_manifest_sha256='c'*64)
    model.eval(); logits=torch.randn(1,1,8,8)
    with torch.no_grad():
        first=model(torch.randn(1,32,8,8),torch.randn(1,256,2,2),logits)
        second=model(torch.randn(1,32,8,8),torch.randn(1,256,2,2),logits)
    assert torch.equal(first,second)
