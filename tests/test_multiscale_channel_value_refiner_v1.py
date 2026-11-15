import copy

import pytest
import torch
from torch.nn import functional as F

from modeling.foreground_conditioned_refiner import _source_sha256
from modeling.multiscale_channel_value_refiner_v1 import (
    MultiscaleChannelValueRefinerV1, channel_matrix,
)
import modeling.multiscale_channel_value_refiner_v1 as module
from scripts.train_region_mixed_loss_v1 import pixel_loss
from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss


def test_channel_matrix_matches_explicit_region_correlation_and_mixes_values():
    q = torch.tensor([[[[1., 2.]], [[3., -1.]]]])
    k = torch.tensor([[[[2., 0.]], [[-1., 1.]]]])
    p = torch.tensor([[[[0.8, 0.2]]]])
    def corr(w):
        a = F.normalize((q * w).flatten(2), dim=-1)
        b = F.normalize((k * w).flatten(2), dim=-1)
        return a @ b.transpose(1, 2)
    actual = channel_matrix(q, k, p, 'foreground_background')
    assert torch.allclose(actual, (corr(p) - corr(1 - p)).softmax(-1))
    values = torch.tensor([[[2., 2.], [0., 0.]]])
    mixed = actual @ values
    assert torch.all(mixed[:, 1] > 0), 'off-diagonal attention must mix value channels'
    diagonal = channel_matrix(q, k, p, 'diagonal')
    assert diagonal[0, 0, 1] == diagonal[0, 1, 0] == 0
    assert torch.equal((diagonal @ values)[:, 1], values[:, 1])


def test_global_and_foreground_matrix_definitions():
    torch.manual_seed(4)
    q, k = torch.randn(2, 4, 3, 2), torch.randn(2, 4, 3, 2)
    p = torch.rand(2, 1, 3, 2)
    def corr(w):
        return F.normalize((q*w).flatten(2), dim=-1) @ F.normalize((k*w).flatten(2), dim=-1).transpose(1, 2)
    assert torch.allclose(channel_matrix(q, k, p, 'global'), corr(torch.ones_like(p)).softmax(-1))
    assert torch.allclose(channel_matrix(q, k, p, 'foreground'), corr(p).softmax(-1))
    assert torch.allclose(channel_matrix(q, k, p, 'foreground').sum(-1), torch.ones(2, 4))
    with pytest.raises(ValueError, match='probabilities'):
        channel_matrix(q, k, p + 2, 'global')


def test_mask_only_cannot_observe_early_or_deep_inputs():
    torch.manual_seed(19)
    net = MultiscaleChannelValueRefinerV1(mode='mask_only', early_channels=2, deep_channels=3, hidden=4)
    torch.nn.init.constant_(net.head[-1].weight, 0.1)
    logits = torch.randn(1, 1, 8, 8)
    a = net(torch.randn(1, 2, 4, 4), torch.randn(1, 3, 2, 2), logits)
    b = net(torch.randn(1, 2, 4, 4), torch.randn(1, 3, 2, 2), logits)
    assert torch.equal(a, b)
    torch.manual_seed(19)
    matched = MultiscaleChannelValueRefinerV1(mode='global', early_channels=2, deep_channels=3, hidden=4)
    matched.load_state_dict(net.state_dict())
    assert torch.equal(a, matched(torch.zeros(1, 2, 4, 4), torch.zeros(1, 3, 2, 2), logits))


def test_two_by_two_matrix_and_contrast_controls_are_independent(monkeypatch):
    torch.manual_seed(42)
    modes=('global','global_contrast','foreground_background_no_contrast','foreground_background')
    reference=MultiscaleChannelValueRefinerV1(mode=modes[0],early_channels=2,deep_channels=3,hidden=4,attention_size=4)
    args=(torch.randn(1,2,5,5),torch.randn(1,3,3,3),torch.randn(1,1,8,8))
    original=module.channel_matrix;matrices={};head_inputs={}
    for mode in modes:
        net=MultiscaleChannelValueRefinerV1(mode=mode,early_channels=2,deep_channels=3,hidden=4,attention_size=4)
        net.load_state_dict(reference.state_dict())
        def collect(q,k,p,m,mode=mode):
            value=original(q,k,p,m);matrices[mode]=value.detach();return value
        monkeypatch.setattr(module,'channel_matrix',collect)
        handle=net.head.register_forward_pre_hook(lambda _m,x,mode=mode:head_inputs.__setitem__(mode,x[0].detach()))
        net(*args);handle.remove()
    assert torch.equal(matrices['global'],matrices['global_contrast'])
    assert torch.equal(matrices['foreground_background_no_contrast'],matrices['foreground_background'])
    assert torch.count_nonzero(head_inputs['global'][:,-2])==0
    assert torch.count_nonzero(head_inputs['foreground_background_no_contrast'][:,-2])==0
    assert torch.equal(head_inputs['global_contrast'][:,-2],head_inputs['foreground_background'][:,-2])
    assert torch.count_nonzero(head_inputs['global_contrast'][:,-2])>0


def test_asinh_arm_changes_only_head_logit_representation_and_preserves_skip():
    torch.manual_seed(8)
    raw=MultiscaleChannelValueRefinerV1(mode='foreground_background',early_channels=2,deep_channels=3,hidden=4,attention_size=4)
    compressed=MultiscaleChannelValueRefinerV1(mode='foreground_background_asinh',early_channels=2,deep_channels=3,hidden=4,attention_size=4)
    compressed.load_state_dict(raw.state_dict())
    logits=torch.tensor([-1024.,-1.,0.,1.,1024.]).reshape(1,1,1,5)
    args=(torch.randn(1,2,2,5),torch.randn(1,3,1,5),logits)
    captured=[]
    h1=raw.head.register_forward_pre_hook(lambda _m,x:captured.append(x[0].detach()))
    h2=compressed.head.register_forward_pre_hook(lambda _m,x:captured.append(x[0].detach()))
    assert torch.equal(raw(*args),logits)
    assert torch.equal(compressed(*args),logits)
    h1.remove();h2.remove()
    assert torch.equal(captured[0][:,:-1],captured[1][:,:-1])
    assert torch.equal(captured[1][:,-1:],torch.asinh(logits))
    z=torch.tensor(0.,requires_grad=True);torch.asinh(z).backward();assert z.grad==1


def test_asinh_reduces_initial_sentinel_head_gradient_in_fixed_synthetic_case():
    torch.manual_seed(2)
    raw=MultiscaleChannelValueRefinerV1(mode='foreground_background',early_channels=2,deep_channels=3,hidden=4,attention_size=4)
    transformed=MultiscaleChannelValueRefinerV1(mode='foreground_background_asinh',early_channels=2,deep_channels=3,hidden=4,attention_size=4)
    transformed.load_state_dict(raw.state_dict())
    args=(torch.randn(1,2,8,8),torch.randn(1,3,4,4),torch.full((1,1,16,16),-1024.))
    target=torch.ones(1,1,16,16)
    for net in (raw,transformed):cmsa_boundary_loss(net(*args),target).backward()
    assert torch.isfinite(raw.head[-1].weight.grad).all()
    assert torch.isfinite(transformed.head[-1].weight.grad).all()
    assert transformed.head[-1].weight.grad.norm()<raw.head[-1].weight.grad.norm()


@pytest.mark.parametrize('mode', MultiscaleChannelValueRefinerV1.MODES)
@pytest.mark.parametrize('native_value', [-1024., 0., 16.])
def test_zero_identity_and_finite_training_gradients_with_sentinel(mode, native_value):
    torch.manual_seed(1)
    net = MultiscaleChannelValueRefinerV1(mode=mode, early_channels=2, deep_channels=3, hidden=4, attention_size=4)
    early = torch.randn(1, 2, 6, 6)
    deep = torch.randn(1, 3, 3, 3)
    logits = torch.full((1, 1, 8, 8), native_value)
    assert torch.equal(net(early, deep, logits), logits)
    # Exercise backward through the attention branch after learning starts.
    torch.nn.init.constant_(net.head[-1].weight, 0.01)
    out = net(early, deep, logits)
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    assert net.query.weight.grad is not None and net.value.weight.grad is not None


def test_modes_have_equal_allocated_parameter_shapes_and_eval_requires_provenance():
    models = [MultiscaleChannelValueRefinerV1(mode=m, early_channels=2, deep_channels=3, hidden=4)
              for m in MultiscaleChannelValueRefinerV1.MODES]
    assert all([tuple(p.shape) for p in x.parameters()] == [tuple(p.shape) for p in models[0].parameters()] for x in models)
    net = models[0].eval()
    args = (torch.zeros(1, 2, 3, 3), torch.zeros(1, 3, 2, 2), torch.zeros(1, 1, 4, 4))
    with pytest.raises(RuntimeError, match='Untrained'):
        net(*args)
    sha = 'a' * 64
    payload = {'format': 'multiscale_channel_value_refiner_v1', 'research_only': True,
               'training_steps': 1, 'source_sha256': _source_sha256(module.__file__), 'mode': net.mode,
               'config': {'early_channels': 2, 'deep_channels': 3, 'hidden': 4, 'attention_size': 32},
               'backbone_sha256': sha, 'feature_pipeline_sha256': sha,
               'train_manifest_sha256': sha, 'state_dict': copy.deepcopy(net.state_dict())}
    provenance = dict(backbone_sha256=sha, feature_pipeline_sha256=sha, train_manifest_sha256=sha)
    net.load_trained_state(payload, **provenance)
    assert torch.equal(net(*args), args[-1])
    bad_geometry = copy.deepcopy(payload)
    bad_geometry['config']['attention_size'] = 16
    with pytest.raises(ValueError, match='geometry'):
        net.load_trained_state(bad_geometry, **provenance)
    with pytest.raises(RuntimeError, match='Untrained'):
        net(*args)
    net.load_trained_state(payload, **provenance)
    payload['mode'] = 'global'
    with pytest.raises(ValueError, match='ablation'):
        net.load_trained_state(payload, **provenance)
    with pytest.raises(RuntimeError, match='Untrained'):
        net(*args)


@pytest.mark.parametrize('present', [False, True])
def test_native_bce_dice_objective_is_finite_for_suppressed_masks(present):
    torch.manual_seed(7)
    net = MultiscaleChannelValueRefinerV1(early_channels=2, deep_channels=3, hidden=4, attention_size=4)
    opt = torch.optim.Adam(net.parameters(), lr=0.0003, weight_decay=0.0001)
    args = (torch.randn(1, 2, 6, 6), torch.randn(1, 3, 3, 3), torch.full((1, 1, 8, 8), -1024.))
    target = torch.full((1, 1, 8, 8), float(present))
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = pixel_loss(net(*args), target, 'bce')
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        if present:
            assert net.head[-1].bias.grad.abs().sum() > 0
        else:
            # Sigmoid(-1024) underflows to zero: an already-correct empty mask
            # contributes no segmentation gradient at exact zero initialization.
            assert net.head[-1].bias.grad.abs().sum() == 0
        opt.step()


@pytest.mark.parametrize('present',[False,True])
def test_boundary_objective_with_suppressed_logits_has_finite_steps(present):
    torch.manual_seed(13)
    net=MultiscaleChannelValueRefinerV1(early_channels=2,deep_channels=3,hidden=4,attention_size=4)
    opt=torch.optim.Adam(net.parameters(),lr=0.0003,weight_decay=0.0001)
    args=(torch.randn(1,2,8,8),torch.randn(1,3,4,4),torch.full((1,1,16,16),-1024.))
    target=torch.full((1,1,16,16),float(present))
    for _ in range(3):
        opt.zero_grad(set_to_none=True); loss=cmsa_boundary_loss(net(*args),target);loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        assert (net.head[-1].bias.grad.abs().sum()>0) == present
        opt.step()
