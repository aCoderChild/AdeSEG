import pytest
import torch
from modeling.lean_recent_spread_exchange import exchange_memory

def test_exchange_preserves_matching_query_condition_and_paired_positions():
    x=torch.arange(16388,dtype=torch.float32).view(-1,1,1);y=x.clone();y[4096:]+=20000
    a={'curr':[x[:3]],'curr_pos':[x[:3]],'memory':x,'memory_pos':x+10,'num_obj_ptr_tokens':4,'return_attn_stat':True}
    b=dict(a,memory=y,memory_pos=y+10)
    for kind,start,end in (('spatial',4096,16384),('newest',12288,16384),('pointers',16384,16388)):
        z=exchange_memory(a,b,kind)
        for k in ('memory','memory_pos'):
            assert torch.equal(z[k][:start],a[k][:start])
            assert torch.equal(z[k][start:end],b[k][start:end])
            assert torch.equal(z[k][end:],a[k][end:])
    changed_query=exchange_memory(a,dict(b,curr=[torch.zeros(3,1,1)]),'spatial')
    assert changed_query['curr'] is a['curr'] and changed_query['curr_pos'] is a['curr_pos']
    with pytest.raises(ValueError,match='Conditioning'):
        bad=y.clone();bad[0]=123;exchange_memory(a,dict(b,memory=bad),'newest')
