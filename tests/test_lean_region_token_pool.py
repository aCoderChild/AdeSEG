import numpy as np
import pytest
import torch
from modeling.cross_frame_token_pool import TokenGroup
from modeling.lean_region_token_pool import boundary_prototype,boundary_scores,region_indices,compact_region_pool,capture_region_scores,RegionPoolRuntime
from test_lean_cross_frame_pool import Model,dense


def test_boundary_prototype_uses_prompt_band_and_not_interior():
    features=torch.zeros(2,16,16);features[0]=1;features[1,4:12,4:12]=2
    mask=np.zeros((16,16),np.float32);mask[4:12,4:12]=1
    p=boundary_prototype(features,mask)
    assert p.shape==(2,) and torch.isclose(p.norm(),torch.tensor(1.)) and p[0]>0 and p[1]>0
    assert boundary_scores(features,p,(4,4)).shape==(16,)
    with pytest.raises(ValueError,match='boundary'):boundary_prototype(features,np.zeros((16,16)))


def test_three_quotas_preserve_distinct_evidence_and_exact_budget():
    g=TokenGroup(1,np.arange(3),np.array([.5,.5,.5]),np.array([1.,-1.,0.]))
    assert region_indices([g],[np.array([-1.,-1.,1.])],3,'three_region')[0].tolist()==[0,1,2]
    assert len(region_indices([g],[np.array([-1.,-1.,1.])],2,'three_region')[0])==2
    with pytest.raises(ValueError):region_indices([g],[np.array([1.,np.nan,0.])],2,'three_region')


def test_compaction_preserves_boundary_feature_position_and_pointer():
    entries=[]
    for frame in (1,2):
        ids=torch.tensor([0,50,100]);v=ids.float().reshape(3,1,1)+frame*1000
        entries.append((frame,{'maskmem_features_seq':v,'maskmem_pos_seq':v+100,'pool_token_ids':ids,
          'pool_usage_rank':torch.tensor([.3,.5,.8]),'pool_contrast':torch.tensor([-1.,0.,1.]),
          'pool_boundary':ids.float()/100,'obj_ptr':torch.tensor([frame])}))
    compact_region_pool(entries,3,'three_region')
    assert sum(e['pool_token_ids'].numel() for _,e in entries if e['pool_token_ids'] is not None)==3
    for frame,e in entries:
        assert e['obj_ptr'].item()==frame
        if e['pool_token_ids'] is not None:
            assert torch.equal(e['pool_boundary'],e['pool_token_ids'].float()/100)
            assert torch.equal(e['maskmem_features_seq'].flatten(),e['pool_token_ids'].float()+frame*1000)
            assert torch.equal(e['maskmem_pos_seq'],e['maskmem_features_seq']+100)


def test_region_hooks_capture_aligned_scores_and_restore():
    model=Model();old=(model._get_image_feature,model.add_new_mask,model._cache_tarr_affinity_from_new_memory,model._dump_retained_tokens_record)
    with capture_region_scores(model,[]):
        model._get_image_feature({},0,1);mask=np.zeros((32,32),bool);mask[:,:12]=True
        model.add_new_mask({},0,1,mask);model._get_image_feature({},1,1)
        entry={'frame_idx':1,'maskmem_features':torch.ones(1,1,64,64)}
        model._cache_tarr_affinity_from_new_memory(entry,{},1,0,False,True,False)
        full=entry['pool_boundary_full'].clone();ids=torch.tensor([4095,0,512]);usage=torch.arange(4096).float()
        model._dump_retained_tokens_record(latest_prev=entry,usage_scores=usage,keep_idx=ids,frame_idx=2)
        assert torch.equal(entry['pool_boundary'],full[ids]) and entry['pool_boundary_full'] is None
    assert (model._get_image_feature,model.add_new_mask,model._cache_tarr_affinity_from_new_memory,model._dump_retained_tokens_record)==old


def test_runtime_admission_uses_full_capacity():
    model=Model();cond=dense(4096);cond['smr_protected']=True;history={}
    for frame in range(1,7):
        e=dense(4096);ids=torch.arange(1024);e.update(smr_state='pending',pending_keep_idx=ids,
          pool_token_ids=ids.clone(),pool_usage_rank=torch.linspace(0,1,1024),pool_contrast=torch.linspace(-1,1,1024),pool_boundary=torch.linspace(1,-1,1024))
        model._compress_memory_entry(e);history[frame]=e
    state={'cond_frame_outputs':{0:cond},'non_cond_frame_outputs':history}
    runtime=RegionPoolRuntime(16384,3072,'three_region');runtime.admit(model,state,{'insurance_mem_bank':{}},0,7,dense(4096))
    plan=[e for e in runtime.events if e['stage']=='admission_plan'][0]
    assert plan['pool_kept']==5120 and plan['pool_candidates']==6144 and plan['planned_reserved_tokens']==16384
