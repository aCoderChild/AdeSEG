import pytest
import torch
from modeling.contrast_refinement_routing import route_refined_logits

@pytest.mark.parametrize('arm,output_refined,memory_refined',[('A',False,False),('B',True,False),('C',False,True),('D',True,True)])
def test_factorial_routes_and_protected_conditioning(arm,output_refined,memory_refined):
    a=torch.zeros(1,1,3,3);b=torch.ones_like(a)
    out,mem=route_refined_logits(a,b,arm,conditioning=False)
    assert torch.equal(out,b if output_refined else a) and torch.equal(mem,b if memory_refined else a)
    out,mem=route_refined_logits(a,b,arm,conditioning=True)
    assert torch.equal(out,a) and torch.equal(mem,a)

def test_incompatible_inputs_fail():
    with pytest.raises(ValueError):route_refined_logits(torch.zeros(1),torch.ones(2),'A',conditioning=False)
    with pytest.raises(ValueError):route_refined_logits(torch.zeros(1),torch.tensor([float('nan')]),'A',conditioning=False)
