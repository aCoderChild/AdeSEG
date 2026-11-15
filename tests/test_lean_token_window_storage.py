import torch
import pytest
import subprocess
import sys
from pathlib import Path
from modeling.lean_token_window_storage import plan_admission,apply_eviction,count_with_insurance,temporal_slot,TokenWindowRuntime

def dense(n):
    return {'maskmem_features':torch.arange(n).reshape(1,1,1,n).float(),
            'maskmem_pos_enc':[torch.arange(n).reshape(1,1,1,n).float()],
            'obj_ptr':torch.ones(1,4),'pred_masks':torch.ones(1,1,2,2),'smr_protected':False}
def test_admission_counts_pending_physically_and_preserves_nonmemory():
    cond=dense(4);old=dense(3);old['pending_keep_idx']=torch.tensor([0]);old['smr_state']='pending'
    latest=dense(4);current=dense(4)
    state={'cond_frame_outputs':{0:cond},'non_cond_frame_outputs':{1:old,2:latest}}
    plan=plan_admission(None,state,3,current,budget=12,insurance_reserve=0)
    assert plan['evict_frames']==[1] and plan['planned_reserved_tokens']==12
    before=old['pred_masks'].clone();ptr=old['obj_ptr'].clone();apply_eviction(state,plan)
    assert old['maskmem_features'] is None and old['maskmem_pos_enc'] is None
    assert torch.equal(old['pred_masks'],before) and torch.equal(old['obj_ptr'],ptr)
    state['non_cond_frame_outputs'][3]=current
    assert count_with_insurance(state,[])['total_tokens']==12

def test_reservation_and_impossible_protected_budget():
    state={'cond_frame_outputs':{0:dense(4)},'non_cond_frame_outputs':{1:dense(2)}}
    with pytest.raises(ValueError):plan_admission(None,state,2,dense(4),budget=9,insurance_reserve=2)
    p=plan_admission(None,state,2,dense(4),budget=11,insurance_reserve=2)
    assert p['evict_frames']==[1] and p['planned_reserved_tokens']==10
    bank=[{'mem_seq':torch.zeros(2,1,1),'pos_seq':torch.zeros(2,1,1)}]
    assert count_with_insurance(state,bank)['insurance_tokens']==2

def test_rejects_reverse_and_misaligned_memory_without_mutating():
    e=dense(2);state={'cond_frame_outputs':{0:dense(4)},'non_cond_frame_outputs':{3:e}}
    with pytest.raises(ValueError):plan_admission(None,state,2,dense(4),budget=12,insurance_reserve=0)
    assert e['maskmem_features'] is not None
    broken=dense(2);broken['maskmem_pos_enc']=[torch.zeros(1,1,1,3)]
    with pytest.raises(ValueError):plan_admission(None,{'cond_frame_outputs':{0:dense(4)}},2,broken,budget=12,insurance_reserve=0)

def test_runtime_reserves_insurance_and_checks_real_write():
    class Model:
        def _compress_memory_entry(self,entry):
            ids=entry['pending_keep_idx'];e=entry['maskmem_features'].flatten(2).permute(2,0,1)
            p=entry['maskmem_pos_enc'][-1].flatten(2).permute(2,0,1)
            entry['maskmem_features_seq']=e[ids];entry['maskmem_pos_seq']=p[ids]
            entry['maskmem_features']=entry['maskmem_pos_enc']=entry['pending_keep_idx']=None
            entry['smr_state']='pruned'
    old=dense(3);old.update(smr_state='pending',pending_keep_idx=torch.tensor([0]))
    state={'cond_frame_outputs':{0:dense(4)},'non_cond_frame_outputs':{1:old,2:dense(4)}}
    bank={'insurance_mem_bank':{0:[{'mem_seq':torch.zeros(2,1,1),'pos_seq':torch.zeros(2,1,1)}]}}
    runtime=TokenWindowRuntime(13,2);current=dense(4)
    plan=runtime.admit(Model(),state,bank,0,3,current)
    assert plan['evict_frames']==[1,2]
    state['non_cond_frame_outputs'][3]=current
    assert runtime.verify(state,bank,0,'after_write',3)['total_tokens']==10
    bank['insurance_mem_bank'][0].append({'mem_seq':torch.zeros(1,1,1),'pos_seq':torch.zeros(1,1,1)})
    with pytest.raises(ValueError,match='Insurance bank exceeds'):runtime.verify(state,bank,0,'after_bank_write',3)

def test_temporal_age_uses_learned_slots_and_clamps_only_oldest():
    assert [temporal_slot(10,10-age,7) for age in range(1,9)]==[6,5,4,3,2,1,1,1]
    with pytest.raises(ValueError):temporal_slot(10,10,7)

def test_author_style_pending_compression_keeps_matching_feature_and_position_indices():
    class Model:
        def _compress_memory_entry(self,entry):
            ids=entry['pending_keep_idx'];feat=entry['maskmem_features'].flatten(2).permute(2,0,1)
            pos=entry['maskmem_pos_enc'][-1].flatten(2).permute(2,0,1)
            entry['maskmem_features_seq']=feat[ids];entry['maskmem_pos_seq']=pos[ids]
            entry['maskmem_features']=entry['maskmem_pos_enc']=entry['pending_keep_idx']=None;entry['smr_state']='pruned'
    old=dense(5);old.update(smr_state='pending',pending_keep_idx=torch.tensor([4,1]))
    state={'cond_frame_outputs':{0:dense(4)},'non_cond_frame_outputs':{1:old}}
    runtime=TokenWindowRuntime(15,2);bank={'insurance_mem_bank':{}}
    runtime.admit(Model(),state,bank,0,2,dense(4))
    assert old['maskmem_features_seq'].flatten().tolist()==[4.,1.]
    assert torch.equal(old['maskmem_features_seq'],old['maskmem_pos_seq'])

def test_isolated_author_compressor_retains_exact_feature_and_position_indices():
    root=Path(__file__).resolve().parents[1]
    code='''
import sys, torch
sys.path.insert(0, sys.argv[1])
from sam2.modeling.sam2_base import SAM2Base
class Model:
    enable_cross_frame_merge=False
    def _memory_bank_log(self, message): pass
    def _profile_tensor_bytes(self, tensor): return tensor.numel()*tensor.element_size()
values=torch.arange(4096).reshape(1,1,64,64).float()
positions=values+10000
idx=torch.tensor([4095,0,500])
entry={'maskmem_features':values,'maskmem_pos_enc':[positions],'pending_keep_idx':idx,'smr_protected':False}
SAM2Base._compress_memory_entry(Model(),entry)
assert entry['maskmem_features'] is None and entry['maskmem_pos_enc'] is None
assert entry['maskmem_features_seq'].flatten().tolist()==[4095.,0.,500.]
assert entry['maskmem_pos_seq'].flatten().tolist()==[14095.,10000.,10500.]
'''
    subprocess.run([sys.executable,'-c',code,str(root/'external/Lean-SAM2-token-window')],cwd=root,check=True,capture_output=True,text=True)
