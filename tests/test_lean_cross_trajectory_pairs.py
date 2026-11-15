import torch
from modeling.lean_cross_trajectory_blocks import BLOCKS
from modeling.lean_cross_trajectory_pairs import PAIRS, swap_pair

def test_each_pair_swaps_both_slices_and_keeps_condition_third_slice_and_pointers():
    mem = torch.arange(12292, dtype=torch.float32).view(-1, 1, 1)
    other = mem.clone(); other[4096:12288] += 20000
    a = {'curr':[mem[:2]], 'curr_pos':[mem[:2]], 'memory':mem,
         'memory_pos':mem+1, 'num_obj_ptr_tokens':4, 'return_attn_stat':True}
    b = dict(a, memory=other, memory_pos=other+1)
    for name, pair in PAIRS.items():
        z = swap_pair(a, b, name)
        for key in ('memory', 'memory_pos'):
            assert torch.equal(z[key][:4096], a[key][:4096])
            assert torch.equal(z[key][12288:], a[key][12288:])
            for block, (start, end) in BLOCKS.items():
                expected = b if block in pair else a
                assert torch.equal(z[key][start:end], expected[key][start:end])
