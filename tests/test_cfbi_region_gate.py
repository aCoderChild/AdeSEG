import torch
import pytest
from modeling.cfbi_region_gate import RegionGate, region_context, author_gate


def test_author_gate_zero_identity_and_trainable():
    torch.manual_seed(1)
    gate = RegionGate()
    x = torch.randn(2, 17, 256)
    context = torch.randn(2, 192)
    assert torch.equal(gate(x, context), x)
    gate(x, context).square().mean().backward()
    assert gate.gate.IA.weight.grad.abs().sum() > 0
    assert gate.gate.IA.bias.grad.abs().sum() > 0
    assert gate.gate.__class__ is author_gate()


def test_grouping_controls_and_empty_region():
    x = torch.arange(6 * 64, dtype=torch.float32).view(6, 64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    context = region_context(x, labels, 'regional').view(3, 64)
    for i in range(3):
        assert torch.equal(context[i], x[labels == i].mean(0))
    assert torch.equal(region_context(x, labels, 'generic').view(3, 64)[0], x.mean(0))
    assert torch.equal(region_context(x, labels, 'shuffled', 15), region_context(x, labels, 'shuffled', 15))
    assert not torch.equal(context.flatten(), region_context(x, labels, 'shuffled', 15).flatten())
    assert torch.equal(region_context(x[:2], labels[:2], 'regional').view(3, 64)[1:], torch.zeros(2, 64))
    with pytest.raises(ValueError):
        region_context(x, labels[:-1], 'regional')


def test_native_memory_alignment_pointer_exclusion_and_framewise_shuffle():
    import hashlib
    from modeling.cfbi_region_gate import NativeCFBIGate
    from modeling.region_bias import feature_key
    a=torch.arange(4096*64,dtype=torch.float32).reshape(4096,64)
    b=a+1e6
    labels_a=torch.arange(4096)%3
    labels_b=(torch.arange(4096)+1)%3
    bank={feature_key(a.unsqueeze(0)):(2,labels_a), feature_key(b.unsqueeze(0)):(5,labels_b)}
    ptr=torch.full((3,64),1e9)
    memory=torch.cat([a,b,ptr]).unsqueeze(1)
    from types import SimpleNamespace
    model=SimpleNamespace(_prepare_memory_conditioned_features=lambda: None)
    reader=NativeCFBIGate(model,RegionGate(),'shuffled','seq2')
    reader.frame=6;reader.bank=bank
    reader.read(None,(),{'memory':memory,'num_obj_ptr_tokens':3})
    shuffled=[]
    for frame,labels in [(2,labels_a),(5,labels_b)]:
        seed=int.from_bytes(hashlib.sha256(f'seq2/{frame}/20261031'.encode()).digest()[:8],'little')%(2**63)
        sample=labels[torch.randperm(len(labels),generator=torch.Generator().manual_seed(seed))]
        assert torch.equal(torch.bincount(sample,minlength=3),torch.bincount(labels,minlength=3))
        shuffled.append(sample)
    assert torch.equal(reader.context,region_context(torch.cat([a,b]),torch.cat(shuffled),'regional'))
    assert reader.trace[-1]['pointer_keys']==3
    assert reader.trace[-1]['memory_frames']==[2,5]
    output=torch.randn(1,7,256)
    assert torch.equal(reader.apply(0,None,(),output),output)
    reader.frame=5
    with pytest.raises(RuntimeError,match='Future memory'):
        reader.read(None,(),{'memory':memory,'num_obj_ptr_tokens':3})
