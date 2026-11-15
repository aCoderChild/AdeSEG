import numpy as np
import pytest
import torch
from modeling.cross_frame_token_pool import TokenGroup,select_global_tokens,tied_percentile,compact_stored_pool,source_score_metadata,apply_compacted_pool

def group(frame,usage,contrast):
    return TokenGroup(frame,np.arange(len(usage)),np.array(usage,float),np.array(contrast,float))

def test_tied_usage_percentiles_are_fixed_and_nonpositional():
    assert np.array_equal(tied_percentile([5,1,5,9]),[.5,0,.5,1])
    assert np.array_equal(tied_percentile([3,3,3]),[.5,.5,.5])

def test_global_pool_selects_across_more_than_five_frames_at_fixed_budget():
    groups=[group(i,[.1,.9],[-.2,.2]) for i in range(8)]
    chosen=select_global_tokens(groups,8,'distributed')
    assert sum(map(len,chosen))==8 and sum(bool(len(x)) for x in chosen)==8
    assert all(np.array_equal(x,[0]) for x in chosen)

def test_usage_and_contrast_differ_and_indices_preserve_order():
    groups=[group(1,[.9,.8],[1.,1.]),group(2,[.7,.6],[-1.,-1.])]
    usage=select_global_tokens(groups,2,'usage');contrast=select_global_tokens(groups,2,'contrast')
    assert usage[0].tolist()==[0,1] and usage[1].tolist()==[]
    assert contrast[0].tolist()==[0] and contrast[1].tolist()==[0]

def test_budget_edge_cases_and_validation():
    g=group(2,[.1,.2],[0,0])
    assert select_global_tokens([g],0,'distributed')[0].size==0
    assert select_global_tokens([g],9,'usage')[0].tolist()==[0,1]
    with pytest.raises(ValueError):select_global_tokens([g,g],1,'usage')
    with pytest.raises(ValueError):select_global_tokens([group(1,[1.2],[0])],1,'usage')
    with pytest.raises(ValueError):tied_percentile([np.nan])

def test_distributed_selection_does_not_follow_author_ranking_order():
    g=TokenGroup(1,np.array([2,0,1]),np.array([.9,.8,.7]),np.zeros(3))
    assert select_global_tokens([g],1,'distributed')[0].tolist()==[1]

def test_transactional_compaction_preserves_feature_position_and_metadata():
    entries=[]
    for frame in (1,2,3):
        ids=torch.tensor([0,100,4095]);v=ids.float().reshape(3,1,1)+frame*10000
        e={'maskmem_features_seq':v,'maskmem_pos_seq':v+50000,'pool_token_ids':ids,
           'pool_usage_rank':torch.tensor([.1,.5,.9]),'pool_contrast':torch.tensor([-1.,0.,1.]),'obj_ptr':torch.tensor([frame])}
        entries.append((frame,e))
    before=[e['maskmem_features_seq'].clone() for _,e in entries]
    result=compact_stored_pool(entries,4,'contrast')
    assert sum(len(x['indices']) for x in result)==4
    assert all(torch.equal(e['maskmem_features_seq'],old) for (_,e),old in zip(entries,before))
    for replacement in result:
        t=replacement['tensors'];ids=t['pool_token_ids'];frame=replacement['frame']
        assert torch.equal(t['maskmem_features_seq'].flatten(),ids.float()+frame*10000)
        assert torch.equal(t['maskmem_pos_seq'],t['maskmem_features_seq']+50000)

def test_bad_metadata_rejects_without_mutation():
    feat=torch.ones(2,1,2);entry={'maskmem_features_seq':feat,'maskmem_pos_seq':torch.ones(2,1,2),
      'pool_token_ids':torch.tensor([0]),'pool_usage_rank':torch.tensor([.2,.3]),'pool_contrast':torch.tensor([0.,0.])}
    with pytest.raises(ValueError):compact_stored_pool([(1,entry)],1,'usage')
    assert entry['maskmem_features_seq'] is feat

def test_source_scores_freeze_original_grid_indices_before_global_compaction():
    usage=torch.arange(4096).float();contrast=torch.linspace(-1,1,4096);keep=torch.tensor([4095,0,2048])
    metadata=source_score_metadata(usage,contrast,keep)
    assert metadata['pool_token_ids'].tolist()==keep.tolist()
    assert metadata['pool_usage_rank'][0]==1 and metadata['pool_usage_rank'][1]==0
    assert torch.equal(metadata['pool_contrast'],contrast[keep])
    with pytest.raises(ValueError):source_score_metadata(usage,contrast,torch.tensor([1,1]))

def test_commit_preserves_prediction_pointers_and_prunes_empty_entries():
    entries=[]
    for frame in (1,2):
        ids=torch.tensor([10,20]);v=ids.float().reshape(2,1,1)
        entry={'maskmem_features_seq':v,'maskmem_pos_seq':v+100,'pool_token_ids':ids,'pool_usage_rank':torch.tensor([.1,.9]),
               'pool_contrast':torch.tensor([0.,0.]),'obj_ptr':torch.tensor([frame]),'pred_masks':torch.ones(1,1,2,2)}
        entries.append((frame,entry))
    replacements=compact_stored_pool(entries,1,'usage');apply_compacted_pool(entries,replacements)
    assert entries[0][1]['pool_token_ids'].tolist()==[20] and entries[1][1]['maskmem_features_seq'] is None
    assert entries[1][1]['obj_ptr'].item()==2 and entries[1][1]['pred_masks'].sum()==4
    assert entries[0][1]['smr_pruned_token_count']==1

def test_online_budget_can_retain_parts_of_eight_source_frames():
    entries=[]
    for frame in range(8):
        ids=torch.arange(1024,dtype=torch.long);feat=ids.float().reshape(-1,1,1)+frame*10000
        entry={'maskmem_features_seq':feat,'maskmem_pos_seq':feat+50000,'pool_token_ids':ids,
               'pool_usage_rank':torch.linspace(0,1,1024),'pool_contrast':torch.linspace(-1,1,1024),
               'obj_ptr':torch.tensor([frame]),'pred_masks':torch.ones(1,1,2,2)}
        entries.append((frame,entry))
        selected=compact_stored_pool(entries,5120,'distributed');apply_compacted_pool(entries,selected)
        entries=[(f,e) for f,e in entries if e['maskmem_features_seq'] is not None]
        assert sum(e['maskmem_features_seq'].shape[0] for _,e in entries)<=5120
    assert len(entries)==8
    for frame,e in entries:
        assert torch.equal(e['maskmem_features_seq'].flatten(),e['pool_token_ids'].float()+frame*10000)
        assert torch.equal(e['maskmem_pos_seq'],e['maskmem_features_seq']+50000)
