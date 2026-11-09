from pathlib import Path
import torch
import pytest
from modeling.rde_value_repair import MemoryValueRepair, author_consistency_loss, value_consistency

RDE = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/references/RDE-VOS-CVPR2022')


def adapter():
    m = MemoryValueRepair(grid_shape=(2,2))
    m.bind_layout([{'frame':0,'start':0,'stop':4},{'frame':3,'start':4,'stop':8}], {0})
    return m


def test_identity_gradients_and_prompt_pointer_preservation():
    m=adapter();x=torch.randn(10,2,64)
    assert torch.equal(m(x,2),x)
    m(x,2).square().mean().backward()
    assert m.mlp.layers[-1].weight.grad.abs().sum()>0
    with torch.no_grad():m.mlp.layers[-1].weight.fill_(.01)
    y=m(x,2)
    assert torch.equal(y[:4],x[:4]) and torch.equal(y[8:],x[8:])
    assert not torch.equal(y[4:8],x[4:8])
    assert torch.equal(m(x[:,:1],2),y[:,:1])
    assert torch.equal(m(x[:,[1,0]],2),y[:,[1,0]])


def test_unbound_or_wrong_layout_rejected():
    m=MemoryValueRepair(grid_shape=(2,2))
    with pytest.raises(RuntimeError,match='bound'):m(torch.randn(8,1,64))
    with pytest.raises(RuntimeError,match='Malformed'):m.bind_layout([{'frame':0,'start':1,'stop':5}],{0})
    m=adapter()
    with pytest.raises(RuntimeError,match='bound'):m(torch.randn(9,1,64))


def test_author_kl_exact_channel_axis_and_teacher_detach():
    criterion=author_consistency_loss(RDE)
    s=torch.randn(8,2,64,requires_grad=True);t=torch.randn(8,2,64,requires_grad=True)
    ids=tuple(range(4,8))
    actual=value_consistency(s,t,ids,criterion)
    expected=criterion(s[4:].permute(1,2,0),t[4:].detach().permute(1,2,0))
    assert torch.equal(actual,expected)
    actual.backward()
    assert t.grad is None and torch.equal(s.grad[:4],torch.zeros_like(s.grad[:4]))
    assert s.grad[4:].abs().sum()>0
