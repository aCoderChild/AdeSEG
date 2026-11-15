import pytest
import torch
from modeling.lean_same_read_fusion import fused_skeleton,replace_fused_skeleton


def test_replacement_changes_only_skeleton_values_and_preserves_pointer_tail():
    torch.manual_seed(3);memory=torch.randn(4110,1,4);pos=torch.randn_like(memory)
    raw=torch.randn(5,1,4);history=torch.randn(1,1,4);fused=fused_skeleton(raw,history,.37);memory[4096:4101]=fused
    source={'start':4096,'raw':raw,'fused':fused}
    out=replace_fused_skeleton(memory,pos,4,source,'raw');noop=replace_fused_skeleton(memory,pos,4,source,'fused')
    assert torch.equal(out[:4096],memory[:4096]) and torch.equal(out[4096:4101],raw)
    assert torch.equal(out[4101:],memory[4101:]) and torch.equal(noop,memory)
    with pytest.raises(ValueError,match='matches'):
        replace_fused_skeleton(memory,pos,4,{'start':4104,'raw':raw,'fused':fused},'raw')
    with pytest.raises(ValueError,match='geometry'):fused_skeleton(raw,history,1.2)
