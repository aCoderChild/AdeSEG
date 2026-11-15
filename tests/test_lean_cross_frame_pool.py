import types
import numpy as np
import pytest
import torch
from modeling.lean_cross_frame_pool import CrossFramePoolRuntime,capture_pool_scores

def dense(n):
    v=torch.arange(n,dtype=torch.float32).reshape(1,1,1,n)
    return {'maskmem_features':v,'maskmem_pos_enc':[v+5000], 'obj_ptr':torch.ones(1,4),'pred_masks':torch.ones(1,1,2,2)}

class Model:
    def _get_image_feature(self,inference_state,frame_idx,batch_size):
        x=torch.ones(1,2,256,256);x[:,1,:,128:]=2
        return None,{'backbone_fpn':[x]}
    def add_new_mask(self,inference_state,frame_idx,obj_id,mask):return 'mask accepted'
    def _cache_tarr_affinity_from_new_memory(self,*args):return 'cache accepted'
    def _dump_retained_tokens_record(self,**kwargs):return 'dump accepted'
    def _compress_memory_entry(self,e):
        ids=e['pending_keep_idx'];feat=e['maskmem_features'].flatten(2).permute(2,0,1);pos=e['maskmem_pos_enc'][-1].flatten(2).permute(2,0,1)
        e['maskmem_features_seq']=feat[ids];e['maskmem_pos_seq']=pos[ids]
        e['maskmem_features']=e['maskmem_pos_enc']=e['pending_keep_idx']=None;e['smr_state']='pruned'

def test_hooks_attach_label_free_scores_matching_author_indices_and_restore():
    model=Model();old=(model._get_image_feature,model.add_new_mask,model._cache_tarr_affinity_from_new_memory,model._dump_retained_tokens_record);events=[]
    with capture_pool_scores(model,events):
        model._get_image_feature({},0,1);mask=np.zeros((32,32),bool);mask[:,:12]=True
        assert model.add_new_mask({},0,1,mask)=='mask accepted'
        model._get_image_feature({},1,1);entry={'frame_idx':1,'maskmem_features':torch.ones(1,1,64,64)}
        assert model._cache_tarr_affinity_from_new_memory(entry,{},1,0,False,True,False)=='cache accepted'
        keep=torch.tensor([4095,0,512]);usage=torch.arange(4096).float()
        assert model._dump_retained_tokens_record(latest_prev=entry,usage_scores=usage,keep_idx=keep,frame_idx=2)=='dump accepted'
        assert entry['pool_token_ids'].tolist()==keep.tolist() and entry['pool_contrast_full'] is None
        assert events[0]['source_frame']==1 and events[0]['tokens']==3
    assert (model._get_image_feature,model.add_new_mask,model._cache_tarr_affinity_from_new_memory,model._dump_retained_tokens_record)==old

def test_runtime_keeps_multiple_history_groups_and_exact_budget():
    model=Model();cond=dense(4096);cond['smr_protected']=True;history={}
    for frame in range(1,8):
        entry=dense(4096);ids=torch.arange(1024,dtype=torch.long);entry.update(smr_state='pending',pending_keep_idx=ids,
          pool_token_ids=ids.clone(),pool_usage_rank=torch.linspace(0,1,1024),pool_contrast=torch.linspace(-1,1,1024))
        model._compress_memory_entry(entry);history[frame]=entry
    # Before admission the runtime must already be within its physical cap.
    # Start with five groups, then admit additional groups one at a time.
    state={'cond_frame_outputs':{0:cond},'non_cond_frame_outputs':{k:history[k] for k in range(1,6)}}
    runtime=CrossFramePoolRuntime(16384,3072,'distributed');bank={'insurance_mem_bank':{}}
    current=dense(4096);runtime.admit(model,state,bank,0,6,current)
    state['non_cond_frame_outputs'][6]=current
    assert runtime.verify(state,bank,0,'after_write',6)['spatial_tokens']==13312
    plan=[x for x in runtime.events if x['stage']=='admission_plan'][0]
    assert plan['pool_budget']==5120 and sum(plan['pool_counts'].values())==5120
    assert all(torch.equal(e['maskmem_pos_seq'],e['maskmem_features_seq']+5000) for k,e in state['non_cond_frame_outputs'].items() if k<6)

def test_runtime_checks_author_selection_against_token_identity():
    entry=dense(4096);ids=torch.arange(1024,dtype=torch.long)
    entry.update(smr_state='pending',pending_keep_idx=ids,pool_token_ids=ids.flip(0),
                 pool_usage_rank=torch.linspace(0,1,1024),pool_contrast=torch.zeros(1024))
    state={'cond_frame_outputs':{0:dense(4096)},'non_cond_frame_outputs':{1:entry}}
    with pytest.raises(ValueError,match='token IDs differ'):
        CrossFramePoolRuntime(16384,3072,'usage').admit(Model(),state,{'insurance_mem_bank':{}},0,2,dense(4096))
