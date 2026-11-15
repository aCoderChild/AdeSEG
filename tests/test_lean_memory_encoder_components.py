import pytest
import torch
from modeling.lean_memory_encoder_components import ARMS,mix_encoder_inputs

def test_all_factorial_combinations_keep_original_inputs_and_geometry():
    def state(x):return {'current_vision_feats':[torch.tensor([x])],'feat_sizes':[(1,1)],'pred_masks_high_res':torch.tensor([x+10]),'object_score_logits':torch.tensor([x+20]),'is_mask_from_pts':False}
    a,b=state(1),state(2)
    assert len(ARMS)==8
    for arm in ARMS:
        z=mix_encoder_inputs(a,b,arm)
        assert z['current_vision_feats'] is (b if arm[0]=='d' else a)['current_vision_feats']
        assert z['pred_masks_high_res'] is (b if arm[1]=='d' else a)['pred_masks_high_res']
        assert z['object_score_logits'] is (b if arm[2]=='d' else a)['object_score_logits']
    with pytest.raises(ValueError):mix_encoder_inputs(a,dict(b,is_mask_from_pts=True),'rdr')
    with pytest.raises(ValueError):mix_encoder_inputs(a,b,'rrx')
