import pytest
import torch
from modeling.lean_temporal_rank_positions import remap_temporal_positions

def test_slots_preserve_spatial_indices_protected_segments_and_multisets():
    grid=torch.arange(64,dtype=torch.float32).reshape(32,2)*10
    offsets=torch.arange(14,dtype=torch.float32).reshape(7,2)/10
    history=[(1,2),(5,2),(12,3)];target=20;insurance=2;pointers=1
    indices=torch.arange(4096+insurance+7,dtype=torch.int64)%32
    pos=torch.zeros(len(indices)+pointers,1,2);pos[:4096+insurance]=17;pos[-1]=29
    cursor=4096+insurance
    for frame,n in history:
        pos[cursor:cursor+n]=(grid[indices[cursor:cursor+n]]+offsets[min(target-frame,6)-1]).unsqueeze(1);cursor+=n
    noop,old=remap_temporal_positions(pos,indices,grid,offsets,target,history,insurance,pointers,'noop')
    ranked,rank=remap_temporal_positions(pos,indices,grid,offsets,target,history,insurance,pointers,'rank')
    shuffled,perm=remap_temporal_positions(pos,indices,grid,offsets,target,history,insurance,pointers,'permuted')
    assert torch.equal(noop,pos) and torch.equal(ranked[:4098],pos[:4098]) and torch.equal(ranked[-1:],pos[-1:])
    assert [x['assigned_slot'] for x in rank]==[2,1,0]
    assert [x['assigned_slot'] for x in perm]==[1,2,0]
    assert sorted(x['assigned_slot'] for x in rank)==sorted(x['assigned_slot'] for x in perm)
    for x in rank:
        assert torch.equal(ranked[x['start']:x['end']],(grid[indices[x['start']:x['end']]]+offsets[x['assigned_slot']]).unsqueeze(1))
    assert torch.equal(pos[-1:],torch.full((1,1,2),29.)) and not torch.equal(ranked,shuffled)
    bad=pos.clone();bad[4098]+=1
    with pytest.raises(ValueError,match='Recorded source'):remap_temporal_positions(bad,indices,grid,offsets,target,history,insurance,pointers,'rank')
    with pytest.raises(ValueError,match='History token ranges'):remap_temporal_positions(pos,indices,grid,offsets,target,history,insurance+1,pointers,'rank')
