import hashlib
import pytest
import torch
from modeling.lean_channel_attention import PromptContrastChannelAttention,author_sha256,load_author_channel_gate
from modeling.lean_early_contrast_anchor import blend_anchor_with_early_contrast


def features():
    torch.manual_seed(21)
    x=torch.randn(32,12,12);fg=torch.nn.functional.normalize(torch.randn(32),dim=0);bg=torch.nn.functional.normalize(torch.randn(32),dim=0)
    return x,fg,bg


def test_author_gate_matches_captured_weights_and_trains():
    torch.manual_seed(1);m=PromptContrastChannelAttention(initialize_for_training=True);x,fg,bg=features()
    score,weights=m(x,fg,bg,(4,4));desc=torch.stack([fg,-bg],-1)[None,:,None,:]
    assert torch.equal(m.gate(desc),desc*weights[None,:,None,None])
    assert score.shape==(16,) and ((weights>0)&(weights<1)).all()
    score.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.gate.parameters())
    with torch.inference_mode(), pytest.raises(RuntimeError,match='Untrained'):
        m(x,fg,bg)
    with torch.no_grad(), pytest.raises(RuntimeError,match='Untrained'):
        m(x,fg,bg)
    m.eval()
    with pytest.raises(RuntimeError,match='Untrained'):m(x,fg,bg)


def test_uniform_channel_weights_recover_existing_contrast():
    m=PromptContrastChannelAttention(initialize_for_training=True);x,fg,bg=features()
    for param in m.gate.parameters():torch.nn.init.zeros_(param)
    score,weights=m(x,fg,bg,(4,4));baseline=blend_anchor_with_early_contrast(torch.zeros(16),x,fg,bg,(4,4),1.)
    assert torch.equal(weights,torch.full((32,),.5))
    assert torch.allclose(score,baseline,atol=1e-6)


def test_checkpoint_interface_and_author_provenance(tmp_path):
    with pytest.raises(ValueError,match='checkpoint'):PromptContrastChannelAttention()
    m=PromptContrastChannelAttention(initialize_for_training=True)
    checkpoint=tmp_path/'checkpoint.pt';backbone='a'*64;pipeline='d'*64
    payload={'format':'sam2_early_prompt_cbam_v1','channels':32,'author_sha256':author_sha256(),'training_steps':1,'gate_state_dict':m.gate.state_dict(),
      'feature_interface':'sam2.1_hiera_base_plus.conv_s0.fpn0','backbone_sha256':backbone,'feature_pipeline_sha256':pipeline,'train_manifest_sha256':'b'*64}
    torch.save(payload,checkpoint);loaded=PromptContrastChannelAttention(checkpoint=checkpoint,expected_backbone_sha256=backbone,expected_feature_pipeline_sha256=pipeline).eval();x,fg,bg=features()
    assert torch.equal(loaded(x,fg,bg)[0],m(x,fg,bg)[0])
    payload['author_sha256']=hashlib.sha256(b'wrong').hexdigest();torch.save(payload,checkpoint)
    with pytest.raises(ValueError,match='metadata'):PromptContrastChannelAttention(checkpoint=checkpoint,expected_backbone_sha256=backbone,expected_feature_pipeline_sha256=pipeline)
    payload['author_sha256']=author_sha256();torch.save(payload,checkpoint)
    with pytest.raises(ValueError,match='backbone'):PromptContrastChannelAttention(checkpoint=checkpoint,expected_backbone_sha256='c'*64,expected_feature_pipeline_sha256=pipeline)
    with pytest.raises(ValueError,match='pipeline'):PromptContrastChannelAttention(checkpoint=checkpoint,expected_backbone_sha256=backbone,expected_feature_pipeline_sha256='e'*64)
    payload['engineering_only']=True;torch.save(payload,checkpoint)
    with pytest.raises(ValueError,match='Engineering fixture'):PromptContrastChannelAttention(checkpoint=checkpoint,expected_backbone_sha256=backbone,expected_feature_pipeline_sha256=pipeline)
    payload['engineering_only']=False
    with pytest.raises(ValueError,match='channels'):PromptContrastChannelAttention(channels=8,initialize_for_training=True)
    with pytest.raises(ValueError,match='channels'):PromptContrastChannelAttention(channels=64,initialize_for_training=True)
    payload['gate_state_dict']['mlp.1.weight'][0,0]=float('nan');torch.save(payload,checkpoint)
    with pytest.raises(ValueError,match='nonfinite'):PromptContrastChannelAttention(checkpoint=checkpoint,expected_backbone_sha256=backbone,expected_feature_pipeline_sha256=pipeline)


def test_adapts_the_real_author_class():
    m=PromptContrastChannelAttention(initialize_for_training=True)
    assert type(m.gate).__name__==load_author_channel_gate().__name__=='ChannelGate'
    assert m.gate.pool_types==['avg','max']

def test_control_gate_descriptors_do_not_replace_contrast_prototypes():
    m=PromptContrastChannelAttention(initialize_for_training=True);x,fg,bg=features()
    zeros=torch.zeros_like(fg)
    score,weights=m(x,fg,bg,(4,4),gate_foreground=zeros,gate_background=zeros)
    weighted_x=torch.nn.functional.normalize(x*weights[:,None,None],dim=0)
    weighted_fg=torch.nn.functional.normalize(fg*weights,dim=0)
    weighted_bg=torch.nn.functional.normalize(bg*weights,dim=0)
    expected=.5*(torch.einsum('chw,c->hw',weighted_x,weighted_fg)-torch.einsum('chw,c->hw',weighted_x,weighted_bg))
    expected=torch.nn.functional.interpolate(expected[None,None],size=(4,4),mode='area').reshape(-1)
    assert torch.allclose(score,expected)
    with pytest.raises(ValueError,match='together'):m(x,fg,bg,gate_foreground=zeros)
