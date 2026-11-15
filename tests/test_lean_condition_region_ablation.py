import numpy as np
import pytest
import torch
from modeling.lean_condition_region_ablation import prompt_region_scores,condition_ablation_indices,delete_condition_tokens

def test_roles_are_deterministic_and_noncondition_and_pointers_preserved():
 mask=np.zeros((64,64),dtype=np.float32);mask[:32,:]=1
 occ=prompt_region_scores(mask);u=np.arange(4096,dtype=float)
 fg=condition_ablation_indices(occ,u,'foreground',1024);bg=condition_ablation_indices(occ,u,'background',1024)
 assert np.all(occ[fg]==1) and np.all(occ[bg]==0) and not set(fg)&set(bg)
 assert np.array_equal(condition_ablation_indices(occ,u,'shuffled',1024),condition_ablation_indices(occ,u,'shuffled',1024))
 assert np.array_equal(condition_ablation_indices(occ,u,'attention_oracle',1024),np.arange(3072,4096))
 memory=torch.arange(4102).reshape(-1,1,1).float();pos=memory+17
 out,p=delete_condition_tokens(memory,pos,2,fg)
 assert len(out)==3078 and torch.equal(out[-6:],memory[-6:]) and torch.equal(p,out+17)
 with pytest.raises(ValueError):delete_condition_tokens(memory,pos,2,np.array([1,1]))
