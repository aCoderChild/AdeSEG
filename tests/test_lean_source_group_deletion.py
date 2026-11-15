import pytest
import torch
from modeling.lean_source_group_deletion import partition_history

def test_source_groups_are_chronological_and_protect_newest():
    src=[{'frame':i,'start':4096+(i-1)*1024,'token_ids':torch.arange(1024)} for i in range(1,6)]
    src.append({'frame':6,'start':9216,'token_ids':torch.arange(4096)})
    groups=partition_history(list(reversed(src)),7,'seq2')
    assert groups==[list(range(4096+i*1024,4096+(i+1)*1024)) for i in range(5)]

def test_unusual_count_excludes_and_overlap_fails():
    src=[{'frame':i,'start':4096+(i-1)*1024,'token_ids':torch.arange(1024)} for i in range(1,6)]
    assert partition_history(src[:4],7,'seq2') is None
    src[1]['start']=src[0]['start']
    with pytest.raises(ValueError):partition_history(src,7,'seq2')
