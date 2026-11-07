"""Identity, author reuse, gradients and isolation before efficacy experiments."""
import copy
import sys

import pytest
import torch

from modeling.complementary_memory.iterative_reader import IterativeMemoryReader


def reader():
    return IterativeMemoryReader(memory_dim=8,channels=32,summaries=4,heads=4,
                                 blocks=3,feedforward=64)


@pytest.fixture
def data():
    torch.set_num_threads(1)
    torch.manual_seed(63)
    return (torch.randn(2,16,32),torch.randn(2,16,32),torch.randn(2,2,8,4,4),
            torch.randn(2,2,8,4,4),torch.rand(2,2,4,4))


def test_exact_native_identity_and_all_blocks_learn(data):
    module=reader()
    assert torch.equal(module(*data,'regional'),data[0])
    optimizer=torch.optim.AdamW(module.parameters(),lr=1e-3)
    module(*data,'regional').square().mean().backward()
    assert module.output_projection.weight.grad.abs().sum()>0
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    result=module(*data,'regional')
    assert not torch.equal(result,data[0])
    result.square().mean().backward()
    for component in [module.position_projection,module.summarizer,*module.reader]:
        gradients=[p.grad for p in component.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients)>0


def test_author_blocks_are_used_and_have_no_attention_mask(data):
    previous=sys.path.copy()
    module=reader(); seen=[]
    assert sys.path==previous
    for block in module.reader:
        assert type(block).__module__=='cutie.model.transformer.object_transformer'
        block.register_forward_pre_hook(lambda m,a,k:seen.append(k['attn_mask']),with_kwargs=True)
    module(*data,'regional')
    assert seen==[None,None,None]


def test_generic_control_is_identical_size_and_ignores_mask(data):
    module=reader(); generic=copy.deepcopy(module)
    with torch.no_grad():
        module.output_projection.weight.normal_(std=.01)
        generic.load_state_dict(module.state_dict())
    assert sum(p.numel() for p in module.parameters())==sum(p.numel() for p in generic.parameters())
    assert torch.equal(generic(*data,'generic'),generic(*data[:-1],1-data[-1],'generic'))
    assert not torch.equal(module(*data,'regional'),module(*data[:-1],1-data[-1],'regional'))


def test_objects_cannot_cross_contaminate(data):
    module=reader().eval()
    with torch.no_grad():
        module.output_projection.weight.normal_(std=.01)
        both=module(*data,'regional')
        reverse=module(*(x.flip(0) for x in data),'regional').flip(0)
        torch.testing.assert_close(both,reverse,atol=1e-6,rtol=1e-5)
        changed=[x.clone() for x in data]; changed[2][1].add_(50)
        altered=module(*changed,'regional')
        assert torch.equal(both[0],altered[0])
        for i in range(2):
            alone=module(*(x[i:i+1] for x in data),'regional')
            torch.testing.assert_close(both[i:i+1],alone,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('value',[0.,1.,.5])
def test_pure_or_tiny_regions_need_no_anchor_threshold(data,value):
    module=reader()
    with torch.no_grad():
        module.output_projection.weight.normal_(std=.01)
    result=module(*data[:-1],torch.full_like(data[-1],value),'regional')
    result.square().mean().backward()
    assert torch.isfinite(result).all()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


def test_exact_resume_after_update(data):
    module=reader(); optimizer=torch.optim.AdamW(module.parameters(),lr=1e-3)
    def step(m,o):
        o.zero_grad(set_to_none=True); loss=m(*data,'regional').square().mean()
        loss.backward(); o.step()
    step(module,optimizer)
    saved=copy.deepcopy({'model':module.state_dict(),'optimizer':optimizer.state_dict(),
                         'rng':torch.random.get_rng_state()})
    step(module,optimizer)
    resumed=reader(); other=torch.optim.AdamW(resumed.parameters(),lr=1e-3)
    resumed.load_state_dict(saved['model']); other.load_state_dict(saved['optimizer'])
    torch.random.set_rng_state(saved['rng']); step(resumed,other)
    for key,value in module.state_dict().items():
        assert torch.equal(value,resumed.state_dict()[key])
