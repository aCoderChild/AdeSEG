import pytest
import torch
from modeling.lean_cross_trajectory_swap import swap_nonconditioning_spatial

def test_swap_preserves_condition_query_pointers_and_pairs_positions():
 mem=torch.arange(4104).reshape(-1,1,1).float();pos=mem+100;other=mem.clone();other[4096:4102]+=10000;otherpos=pos.clone();otherpos[4096:4102]+=20000
 x={'curr':[torch.ones(2,1,1)],'curr_pos':[torch.zeros(2,1,1)],'memory':mem,'memory_pos':pos,'num_obj_ptr_tokens':2,'return_attn_stat':True}
 y={**x,'memory':other,'memory_pos':otherpos}
 out=swap_nonconditioning_spatial(x,y)
 assert torch.equal(out['memory'][:4096],mem[:4096]) and torch.equal(out['memory'][4096:4102],other[4096:4102]) and torch.equal(out['memory'][-2:],mem[-2:])
 assert torch.equal(out['memory_pos'][4096:4102],otherpos[4096:4102]) and torch.equal(out['memory_pos'][-2:],pos[-2:])
 assert torch.equal(mem,torch.arange(4104).reshape(-1,1,1).float())
 with pytest.raises(ValueError,match='query'):
  swap_nonconditioning_spatial(x,{**y,'curr':[torch.zeros(2,1,1)]})
 with pytest.raises(ValueError,match='Conditioning'):
  changed=other.clone();changed[0]=9;swap_nonconditioning_spatial(x,{**y,'memory':changed})
 with pytest.raises(ValueError,match='geometry'):
  swap_nonconditioning_spatial(x,{**y,'num_obj_ptr_tokens':1})
