import pytest
import torch

from modeling.lean_mps_decoder_observer import observe_decoder_reads


class Dummy:
    device=torch.device('mps')
    fill_hole_area=0

    def _get_image_feature(self,state,frame_idx,batch_size):
        return None,{'backbone_fpn':[state['early']]},None,None,None

    def _forward_sam_heads(self,**kwargs):
        low=torch.zeros(1,1,256,256);high=torch.zeros(1,1,1024,1024)
        return low,high,torch.ones(1,1),low,high,torch.zeros(1,256),torch.ones(1,1)


def test_observer_is_read_only_and_restores_hooks(tmp_path):
    model=Dummy();state={'early':torch.randn(1,32,256,256),'video_height':9,'video_width':11}
    observations=[];fg=torch.randn(32);bg=torch.randn(32)
    expected=model._forward_sam_heads(high_res_features=[state['early']])
    with observe_decoder_reads(model,{3},tmp_path,(fg,bg),observations):
        model._get_image_feature(state,3,1)
        actual=model._forward_sam_heads(high_res_features=[state['early']])
        assert all(torch.equal(a,b) for a,b in zip(expected,actual,strict=True))
    assert len(observations)==1 and observations[0]['frame']==3
    f=torch.load(tmp_path/'features'/'000003.pt',map_location='cpu',weights_only=True)
    s=torch.load(tmp_path/'snapshots'/'000003.pt',map_location='cpu',weights_only=True)
    assert torch.equal(f['feature'],state['early'][0]) and torch.equal(s['decoder_inputs']['high_res_features'][0],state['early'])
    assert (s['video_height'],s['video_width'])==(9,11)
    assert '_get_image_feature' not in vars(model) and '_forward_sam_heads' not in vars(model)


def test_observer_rejects_mismatch_and_restores_hooks(tmp_path):
    model=Dummy();state={'early':torch.randn(1,32,256,256),'video_height':9,'video_width':11}
    with pytest.raises(ValueError,match='decoder input'):
        with observe_decoder_reads(model,{3},tmp_path,(torch.ones(32),torch.ones(32)),[]):
            model._get_image_feature(state,3,1)
            model._forward_sam_heads(high_res_features=[torch.zeros_like(state['early'])])
    assert '_get_image_feature' not in vars(model) and '_forward_sam_heads' not in vars(model)
