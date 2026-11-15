import torch
import pytest
from modeling.lean_token_storage_observer import representation,snapshot

def test_pending_is_physically_dense_despite_logical_selection():
    e={'maskmem_features':torch.zeros(1,4,4,4),'maskmem_pos_enc':[torch.zeros(1,4,4,4)],
       'pending_keep_idx':torch.arange(4),'smr_state':'pending'}
    r=representation(e)
    assert r['feature_tokens']==16 and r['position_tokens']==16 and r['pending_indices']==4
    class Model:
        def _profile_memory_entry(self,entry):return {'actual_tokens':4}
    s=snapshot(Model(),{'cond_frame_outputs':{},'non_cond_frame_outputs':{5:e}},None,None)
    assert s['spatial_feature_tokens']==16 and s['logical_spatial_tokens']==4

def test_compressed_and_invalid_position_counts():
    e={'maskmem_features_seq':torch.zeros(3,1,4),'maskmem_pos_seq':torch.zeros(3,1,4)}
    assert representation(e)['feature_tokens']==3
    e['maskmem_pos_seq']=torch.zeros(4,1,4)
    with pytest.raises(ValueError):representation(e)

def test_snapshot_reports_insurance_and_pointers_separately():
    entry={'maskmem_features_seq':torch.zeros(3,1,4),
           'maskmem_pos_seq':torch.zeros(3,1,4),'obj_ptr':torch.zeros(1,4)}
    class Model:
        def _profile_memory_entry(self,entry):return {'actual_tokens':3}
    state={'insurance_mem_bank':{0:[{'mem_seq':torch.zeros(2,1,4),'pos_seq':torch.zeros(2,1,4)}]}}
    s=snapshot(Model(),{'cond_frame_outputs':{},'non_cond_frame_outputs':{1:entry}},state,0)
    assert s['spatial_feature_tokens']==3 and s['insurance_feature_tokens']==2
    assert s['object_pointer_elements']==4 and s['insurance_entries']==[2]
    state['insurance_mem_bank'][0][0]['pos_seq']=torch.zeros(1,1,4)
    with pytest.raises(ValueError):snapshot(Model(),{'non_cond_frame_outputs':{1:entry}},state,0)
