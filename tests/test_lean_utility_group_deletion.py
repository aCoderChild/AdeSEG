import torch
import pytest
from modeling.lean_utility_group_deletion import partition_history,delete_group


def test_partition_disjoint_deterministic_and_protects_newest():
    sources=[{'frame':i,'start':4096+(i-1)*1024,'token_ids':torch.arange(1024)} for i in range(1,6)]
    sources.append({'frame':6,'start':9216,'token_ids':torch.arange(4096)})
    groups=partition_history(sources,7,'seq2')
    assert groups==partition_history(sources,7,'seq2') and len(groups)==5
    assert len({x for group in groups for x in group})==5120
    assert max(x for group in groups for x in group)<9216
    memory=torch.arange((4096+5120+4096+4)*2).reshape(-1,1,2)
    result,pos=delete_group(memory,memory.clone(),4,groups[0])
    assert len(result)==len(memory)-1024 and torch.equal(result,pos)
    assert torch.equal(result[-4100:],memory[-4100:])


def test_partition_rejects_overlap_and_deletion_rejects_pointer():
    sources=[{'frame':1,'start':4096,'token_ids':torch.arange(1024)}]*2
    with pytest.raises(ValueError):partition_history(sources,4,'seq2')
    memory=torch.zeros(6000,1,2)
    with pytest.raises(ValueError):delete_group(memory,memory,4,list(range(4973,5997)))
