import pytest
import torch
from modeling.bounded_anchor_raw_refiner import BoundedAnchorRawRefiner


def tensors():
    return (torch.randn(1,32,4,4),torch.randn(1,256,2,2),torch.tensor([[[[-1024.,2.],[4.,-3.]]]]),torch.randn(1,3,2,2),torch.rand(1,3))


def test_controls_are_same_capacity_and_start_at_bounded_anchor():
    args=tensors()
    for mode in ('raw','zero','spatial_shift'):
        torch.manual_seed(8);model=BoundedAnchorRawRefiner(raw_mode=mode).train()
        assert sum(p.numel() for p in model.parameters())==48453
        output=model(*args)
        torch.testing.assert_close(output,16*torch.tanh(args[2]/16))
        assert torch.isfinite(output).all()


def test_requires_raw_inputs_and_specific_trained_checkpoint():
    args=tensors();model=BoundedAnchorRawRefiner().train()
    with pytest.raises(RuntimeError):model(*args[:3])
    with pytest.raises(ValueError):model.load_trained_state({'format':'raw_multimask_refiner_v1'},backbone_sha256='0'*64,feature_pipeline_sha256='0'*64,train_manifest_sha256='0'*64,raw_manifest_sha256='0'*64)
    model.eval()
    with pytest.raises(RuntimeError):
        with torch.no_grad():model(*args)


def test_suppressed_positive_has_finite_trainable_gradient():
    args=tensors();model=BoundedAnchorRawRefiner().train()
    out=model(*args);loss=torch.nn.functional.binary_cross_entropy_with_logits(out,torch.ones_like(out));loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.head[-1].bias.grad.abs().sum()>0


def test_valid_checkpoint_roundtrip_and_provenance_guard():
    from pathlib import Path
    from modeling.foreground_conditioned_refiner import _source_sha256
    from modeling.lean_channel_attention import author_sha256
    import modeling.bounded_anchor_raw_refiner as module
    here=Path(module.__file__).parent
    model=BoundedAnchorRawRefiner(raw_mode='raw')
    payload={'format':'bounded_anchor_raw_refiner_v1','research_only':True,'training_steps':1,
             'bounded_source_sha256':_source_sha256(module.__file__),'raw_model_sha256':_source_sha256(here/'raw_multimask_refiner.py'),
             'conditioned_source_sha256':_source_sha256(here/'foreground_conditioned_refiner.py'),'base_source_sha256':_source_sha256(here/'multilevel_contrast_refiner.py'),
             'author_sha256':author_sha256(),'anchor_scale':16.,'raw_mode':'raw','gate_mode':'region','contrast_head':True,
             'backbone_sha256':'1'*64,'feature_pipeline_sha256':'2'*64,'train_manifest_sha256':'3'*64,'raw_manifest_sha256':'4'*64,
             'state_dict':model.state_dict()}
    identity={'backbone_sha256':'1'*64,'feature_pipeline_sha256':'2'*64,'train_manifest_sha256':'3'*64,'raw_manifest_sha256':'4'*64}
    model.load_trained_state(payload,**identity);model.eval()
    with torch.no_grad():assert torch.isfinite(model(*tensors())).all()
    payload['anchor_scale']=8.
    with pytest.raises(ValueError):model.load_trained_state(payload,**identity)


def test_explicit_float32_input_and_output_contract():
    model=BoundedAnchorRawRefiner().train();args=tensors()
    assert model(*args).dtype==torch.float32
    for i in range(len(args)):
        altered=list(args);altered[i]=altered[i].half()
        with pytest.raises(ValueError,match='float32'):
            model(*altered)
