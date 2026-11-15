import pytest
import torch
from modeling.lean_cross_trajectory_blocks import BLOCKS, swap_block

def test_block_swaps_keys_and_positions_together_without_touching_other_keys():
    x = torch.arange(12292, dtype=torch.float32).view(-1, 1, 1)
    y = x.clone(); y[4096:12288] += 20000
    a = {'curr':[x[:2]], 'curr_pos':[x[:2]], 'memory':x, 'memory_pos':x+1, 'num_obj_ptr_tokens':4, 'return_attn_stat':True}
    b = dict(a, memory=y, memory_pos=y+1)
    for block,(start,end) in BLOCKS.items():
        z=swap_block(a,b,block)
        for key in ('memory','memory_pos'):
            assert torch.equal(z[key][:start],a[key][:start])
            assert torch.equal(z[key][start:end],b[key][start:end])
            assert torch.equal(z[key][end:],a[key][end:])
    with pytest.raises(ValueError):
        swap_block(a,dict(b,num_obj_ptr_tokens=5),'first')
