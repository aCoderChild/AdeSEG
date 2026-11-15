import numpy as np
import pytest
import torch
from modeling.cross_frame_token_pool import TokenGroup
from modeling.lean_region_read_intervention import remove_ranked_tokens,locate_verified_span


def test_matched_interventions_preserve_protected_and_pointer_tokens():
    mem=torch.arange(16).float().reshape(16,1,1);pos=mem+100
    g=TokenGroup(2,np.arange(6),np.linspace(0,1,6),np.linspace(-1,1,6))
    segment={'start':4,'group':g,'boundary':np.linspace(1,-1,6)}
    for method in ('usage','contrast','distributed','three_region','foreground','shuffled'):
        out,pp,ids=remove_ranked_tokens(mem,pos,2,[segment],[(0,4),(10,14)],method,3,seed=4)
        assert len(ids)==3 and len(out)==13 and all(4<=i<10 for i in ids)
        assert torch.equal(out[:4],mem[:4]) and torch.equal(out[-6:],mem[-6:]) and torch.equal(pp,out+100)
        assert torch.equal(out,mem[[i for i in range(16) if i not in ids]])
    with pytest.raises(ValueError,match='overlap'):
        remove_ranked_tokens(mem,pos,2,[segment],[(0,5)],'usage',3)
    with pytest.raises(ValueError,match='enough'):
        remove_ranked_tokens(mem,pos,2,[segment],[(0,4)],'usage',7)

def test_source_span_requires_matching_features_and_temporal_positions():
    mem=torch.arange(16).float().reshape(16,1,1);pos=mem+100
    source=mem[4:8].clone();source_pos=pos[4:8]-3
    offsets=torch.tensor([[0.],[3.],[5.]])
    assert locate_verified_span(mem,pos,2,source,source_pos,offsets,[(0,4),(10,14)])==4
    with pytest.raises(ValueError,match='unique'):
        locate_verified_span(mem,pos,2,source,source_pos,torch.tensor([[0.]]),[(0,4)])
    duplicated=torch.cat((mem[:8],source,mem[12:]),dim=0)
    dup_pos=torch.cat((pos[:8],pos[4:8],pos[12:]),dim=0)
    with pytest.raises(ValueError,match='unique'):
        locate_verified_span(duplicated,dup_pos,2,source,source_pos,offsets,[(0,4)])
