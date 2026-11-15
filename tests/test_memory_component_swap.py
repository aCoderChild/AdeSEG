import torch
import pytest
from modeling.native_history_cache import hybrid_memory_outputs


def test_component_swap_pairs_spatial_position_and_leaves_metadata_and_sources():
    base={'cond_frame_outputs':{0:{'maskmem_features':torch.tensor([1]),
          'maskmem_pos_enc':[torch.tensor([2])],'obj_ptr':torch.tensor([3]),
          'pred_masks':torch.tensor([4]),'object_score_logits':torch.tensor([5])}},
          'non_cond_frame_outputs':{}}
    donor={'cond_frame_outputs':{0:{k:([torch.tensor([20])] if k=='maskmem_pos_enc' else torch.tensor([10]))
                                     for k in base['cond_frame_outputs'][0]}},'non_cond_frame_outputs':{}}
    spatial=hybrid_memory_outputs(base,donor,component='spatial')['cond_frame_outputs'][0]
    assert spatial['maskmem_features'] is donor['cond_frame_outputs'][0]['maskmem_features']
    assert spatial['maskmem_pos_enc'] is donor['cond_frame_outputs'][0]['maskmem_pos_enc']
    assert spatial['obj_ptr'] is base['cond_frame_outputs'][0]['obj_ptr']
    pointer=hybrid_memory_outputs(base,donor,component='pointer')['cond_frame_outputs'][0]
    assert pointer['obj_ptr'] is donor['cond_frame_outputs'][0]['obj_ptr']
    assert pointer['pred_masks'] is base['cond_frame_outputs'][0]['pred_masks']
    assert base['cond_frame_outputs'][0]['maskmem_features'].item()==1
    with pytest.raises(ValueError,match='same bucket/frame'):
        hybrid_memory_outputs(base,{},component='spatial')
