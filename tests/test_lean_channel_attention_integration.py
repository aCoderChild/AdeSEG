import numpy as np
import pytest
import torch
from modeling.lean_channel_attention import PromptContrastChannelAttention,author_sha256
from modeling.lean_channel_attention_integration import capture_channel_attention_scores
from test_lean_cross_frame_pool import Model


class EarlyModel(Model):
    def _get_image_feature(self,inference_state,frame_idx,batch_size):
        _,out=super()._get_image_feature(inference_state,frame_idx,batch_size)
        out['backbone_fpn'][0]=out['backbone_fpn'][0].repeat(1,16,1,1)
        return None,out


def test_author_channel_gate_observer_uses_same_token_indices_and_restores(tmp_path):
    # This metadata and all-zero weights are synthetic engineering fixtures,
    # not a trained checkpoint and never used for an efficacy experiment.
    original=PromptContrastChannelAttention(initialize_for_training=True)
    for p in original.parameters():torch.nn.init.zeros_(p)
    backbone='a'*64;pipeline='d'*64;path=tmp_path/'fixture.pt'
    torch.save({'format':'sam2_early_prompt_cbam_v1','channels':32,'author_sha256':author_sha256(),'training_steps':1,
      'feature_interface':'sam2.1_hiera_base_plus.conv_s0.fpn0','backbone_sha256':backbone,'feature_pipeline_sha256':pipeline,'train_manifest_sha256':'b'*64,
      'gate_state_dict':original.gate.state_dict()},path)
    model=EarlyModel();old=(model._get_image_feature,model.add_new_mask,model._cache_tarr_affinity_from_new_memory,model._dump_retained_tokens_record);events=[]
    with capture_channel_attention_scores(model,path,backbone,pipeline,events):
        model._get_image_feature({},0,1);mask=np.zeros((32,32),bool);mask[:,:12]=True
        model.add_new_mask({},0,1,mask);model._get_image_feature({},1,1)
        entry={'frame_idx':1,'maskmem_features':torch.ones(1,1,64,64)}
        model._cache_tarr_affinity_from_new_memory(entry,{},1,0,False,True,False)
        scores=entry['pool_contrast_full'].clone();ids=torch.tensor([4095,0,512]);usage=torch.arange(4096).float()
        model._dump_retained_tokens_record(latest_prev=entry,usage_scores=usage,keep_idx=ids,frame_idx=2)
        assert torch.equal(entry['pool_contrast'],scores[ids]) and entry['pool_contrast_full'] is None
        assert len(events)==1 and len(events[0]['channel_weights_sha256'])==64 and len(events[0]['channel_contrast_sha256'])==64
    assert (model._get_image_feature,model.add_new_mask,model._cache_tarr_affinity_from_new_memory,model._dump_retained_tokens_record)==old

    with pytest.raises(RuntimeError,match='forced interruption'):
        with capture_channel_attention_scores(model,path,backbone,pipeline,[]):
            raise RuntimeError('forced interruption')
    assert (model._get_image_feature,model.add_new_mask,model._cache_tarr_affinity_from_new_memory,model._dump_retained_tokens_record)==old
