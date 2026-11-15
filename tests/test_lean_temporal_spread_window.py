import pytest
import torch
from modeling.lean_temporal_spread_window import select_spread,plan_spread
from modeling.lean_token_window_storage import apply_eviction

def test_temporal_spread_respects_budget_and_tie_rule():
    selected,evicted,used=select_spread(0,4096,[(1,1024),(2,1024),(3,1024),(4,1024),(5,1024),(6,1024),(7,4096)],16384,3072)
    assert evicted==[6] and [x[0] for x in selected]==[1,2,3,4,5,7] and used==16384
    selected,evicted,used=select_spread(0,4096,[(1,1024),(2,1024),(3,1024),(4,1024),(5,1024),(7,1024),(8,4096)],16384,3072)
    assert evicted==[4] and [x[0] for x in selected]==[1,2,3,5,7,8] and used==16384
    with pytest.raises(ValueError): select_spread(0,4096,[(1,4096)],10000,3072)
    with pytest.raises(ValueError): select_spread(2,4096,[(1,1024),(3,4096)],16384,3072)

def test_runtime_plan_preserves_positions_and_pointers_during_eviction():
    def entry(n):return {'maskmem_features_seq':torch.ones(n,1,64),'maskmem_pos_seq':torch.ones(n,1,64),'obj_ptr':torch.ones(1,1,256),'smr_state':'stored'}
    state={'cond_frame_outputs':{0:entry(4096)},'non_cond_frame_outputs':{i:entry(1024) for i in range(1,7)}}
    plan=plan_spread(state,7,entry(4096),16384,3072);assert plan['evict_frames']==[6]
    ptr=state['non_cond_frame_outputs'][6]['obj_ptr'].clone();apply_eviction(state,plan)
    assert state['non_cond_frame_outputs'][6]['maskmem_features_seq'] is None
    assert state['non_cond_frame_outputs'][6]['maskmem_pos_seq'] is None
    assert torch.equal(state['non_cond_frame_outputs'][6]['obj_ptr'],ptr)
    assert state['cond_frame_outputs'][0]['maskmem_features_seq'].shape[0]==4096

def test_streaming_eviction_really_keeps_nonrecent_entries_without_extra_tokens():
    history=[]
    for frame in range(1,31):
        history=[(i,1024) for i,_ in history]+[(frame,4096)]
        history,_,used=select_spread(0,4096,history,16384,3072)
        assert used<=16384 and history[-1][0]==frame
    assert len(history)==6
    assert [i for i,_ in history] != list(range(25,31))
    assert min(i for i,_ in history)<25
