"""CPU smoke tests of the clean scientific research interfaces."""
import torch
from modeling.feature_aggregation import MultistageFeatureAggregator
from adeseg.losses.foreground_preservation import positive_teacher_penalty, positive_preservation_penalty

def test_feature_aggregator_preserves_native_logits_at_initialization():
    torch.manual_seed(9)
    model=MultistageFeatureAggregator(early_channels=2,deep_channels=4,hidden=8,attention_size=4)
    e=torch.randn(1,2,6,6)
    d=torch.randn(1,4,3,3)
    native=torch.randn(1,1,8,8)
    assert torch.equal(model(e,d,native),native)

def test_teacher_preservation_is_zero_on_unchanged_teacher():
    native=torch.tensor([[[[2.,0.],[-2.,-1024.]]]])
    target=torch.ones_like(native)
    assert positive_teacher_penalty(native,native,target).item()==0
    assert positive_preservation_penalty(native,native,target).item()==0
