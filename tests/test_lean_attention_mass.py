import pytest
import torch
from modeling.lean_attention_mass import summarize_key_usage

def test_exact_mass_partition_and_empty_pointer():
    x=torch.tensor([.1,.2,.3,.4]);z=summarize_key_usage(x,2,1)
    assert z['conditioning_mass']==pytest.approx(.3)
    assert z['other_spatial_mass']==pytest.approx(.3)
    assert z['pointer_mass']==pytest.approx(.4)
    assert summarize_key_usage(x,2,0)['pointer_mass']==0

def test_invalid_segments_and_nonprobability_inputs_fail():
    with pytest.raises(ValueError):summarize_key_usage(torch.ones(3),2,2)
    with pytest.raises(ValueError):summarize_key_usage(torch.tensor([0.,0.]),1,0)
    with pytest.raises(ValueError):summarize_key_usage(torch.tensor([-.1,.2]),1,0)
