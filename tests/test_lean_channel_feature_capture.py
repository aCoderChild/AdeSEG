import numpy as np
import pytest
import torch
from modeling.lean_channel_feature_capture import capture_channel_features
from test_lean_cross_frame_pool import Model


class EarlyModel(Model):
    def _get_image_feature(self,inference_state,frame_idx,batch_size):
        _,out=super()._get_image_feature(inference_state,frame_idx,batch_size)
        out['backbone_fpn'][0]=out['backbone_fpn'][0].repeat(1,16,1,1)
        return None,out


def test_captures_once_after_automatic_prompt_and_restores(tmp_path):
    model=EarlyModel();old=(model._get_image_feature,model.add_new_mask);events=[]
    with capture_channel_features(model,{1},tmp_path,events):
        model._get_image_feature({},0,1);mask=np.zeros((32,32),bool);mask[:,:12]=True
        model.add_new_mask({},0,1,mask);model._get_image_feature({},1,1);model._get_image_feature({},1,1)
    sample=torch.load(tmp_path/'features/000001.pt',weights_only=True)
    assert len(events)==1 and sample['feature'].shape==(32,256,256) and sample['foreground'].shape==(32,)
    assert (model._get_image_feature,model.add_new_mask)==old


def test_missing_automatic_prompt_fails_and_restores(tmp_path):
    model=EarlyModel();old=(model._get_image_feature,model.add_new_mask)
    with pytest.raises(ValueError,match='before automatic'):
        with capture_channel_features(model,{1},tmp_path,[]):model._get_image_feature({},1,1)
    assert (model._get_image_feature,model.add_new_mask)==old
