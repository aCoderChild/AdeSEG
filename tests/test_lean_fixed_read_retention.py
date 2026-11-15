import numpy as np
import pytest
import torch
from modeling.cross_frame_token_pool import TokenGroup, select_global_tokens
from modeling.lean_fixed_read_retention import retain_ranked_tokens
from modeling.lean_region_read_intervention import remove_ranked_tokens


def test_retention_is_complement_of_existing_ranked_selection_and_preserves_positions():
    mem=torch.arange(18).reshape(18,1,1).float();pos=mem+100
    group=TokenGroup(2,np.arange(8),np.linspace(0,1,8),np.linspace(-1,1,8))
    segments=[{'start':4,'group':group,'boundary':np.linspace(1,-1,8)}]
    for method in ('usage','contrast','distributed','three_region','foreground','shuffled_contrast'):
        out,p,kept,removed=retain_ranked_tokens(mem,pos,2,segments,[(0,4),(12,16)],method,3,seed=4)
        assert len(kept)==3 and len(removed)==5
        assert set(kept)|set(removed)==set(range(4,12)) and not set(kept)&set(removed)
        assert torch.equal(out,mem[[i for i in range(18) if i not in removed]]) and torch.equal(p,out+100)
        if method!='shuffled_contrast':
            _,_,selected=remove_ranked_tokens(mem,pos,2,segments,[(0,4),(12,16)],method,3,seed=4)
            assert kept==selected
        full,fullpos,_,removed=retain_ranked_tokens(mem,pos,2,segments,[(0,4),(12,16)],method,8,seed=4)
        assert not removed and torch.equal(full,mem) and torch.equal(fullpos,pos)
    with pytest.raises(ValueError,match='overlap'):
        retain_ranked_tokens(mem,pos,2,segments,[(0,5)],'usage',3)


def test_shuffled_control_keeps_usage_and_only_permutes_contrast():
    memory=torch.arange(18).reshape(18,1,1).float();positions=memory+100
    group=TokenGroup(2,np.arange(8),np.linspace(0,1,8),np.linspace(-1,1,8))
    segment={'start':4,'group':group,'boundary':np.zeros(8)}
    _,_,chosen,_=retain_ranked_tokens(memory,positions,2,[segment],[(0,4),(12,16)],'shuffled_contrast',3,seed=4)
    permutation=np.random.default_rng(4).permutation(8)
    shuffled=TokenGroup(group.frame,group.token_ids,group.usage_rank,group.contrast[permutation])
    expected=select_global_tokens([shuffled],3,'contrast')[0]
    assert chosen==[4+int(i) for i in expected]
