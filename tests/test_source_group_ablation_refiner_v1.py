import hashlib

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from modeling.foreground_conditioned_refiner import ForegroundConditionedRefiner
from modeling.source_group_ablation_refiner_v1 import ARMS, SourceGroupAblationRefiner, spatial_shuffle
from scripts.source_group_ablation_inputs_v1 import read_rgb


def _inputs():
    torch.manual_seed(109)
    return torch.randn(1, 32, 8, 8), torch.randn(1, 256, 4, 4), torch.randn(1, 1, 8, 8)


def _model(arm):
    torch.manual_seed(47)
    model = SourceGroupAblationRefiner(control=arm).train()
    # The author's zero-initialized last layer would hide all input changes.
    torch.nn.init.normal_(model.head[-1].weight, std=0.01)
    return model


def test_reference_matches_existing_refiner_and_capacity():
    torch.manual_seed(47)
    existing = ForegroundConditionedRefiner(gate_mode='region').train()
    candidate = _model('reference')
    existing.load_state_dict(candidate.state_dict(), strict=True)
    early, deep, logits = _inputs()
    torch.testing.assert_close(existing(early, deep, logits), candidate(early, deep, logits), rtol=0, atol=0)
    assert sum(p.numel() for p in candidate.parameters()) == 48005
    for arm in ARMS:
        assert [(k, tuple(v.shape)) for k, v in _model(arm).state_dict().items()] == [
            (k, tuple(v.shape)) for k, v in candidate.state_dict().items()]


def test_single_scale_controls_ignore_only_the_ablated_input():
    early, deep, logits = _inputs()
    early_model = _model('early_only')
    deep_model = _model('deep_only')
    torch.testing.assert_close(early_model(early, deep, logits), early_model(early, deep + 31, logits), rtol=0, atol=0)
    torch.testing.assert_close(deep_model(early, deep, logits), deep_model(early - 31, deep, logits), rtol=0, atol=0)
    assert not torch.equal(early_model(early, deep, logits), early_model(early + 4, deep, logits))
    assert not torch.equal(deep_model(early, deep, logits), deep_model(early, deep + 4, logits))


def test_shuffle_is_deterministic_bijective_and_shared_across_channels():
    features = torch.arange(48).reshape(1, 1, 6, 8).float()
    stacked = torch.cat([features, features + 100], dim=1)
    shuffled = spatial_shuffle(stacked)
    assert not torch.equal(shuffled, stacked)
    assert torch.equal(shuffled, spatial_shuffle(stacked))
    assert torch.equal(shuffled[:, 1] - shuffled[:, 0], torch.full((1, 6, 8), 100.0))
    assert torch.equal(shuffled[:, 0].flatten().sort().values, features.flatten().sort().values)


def test_foreground_only_drops_background_descriptor_and_contrast():
    model = _model('foreground_only')
    early, deep, logits = _inputs()
    observed = {}
    original = model.channel_weights
    def record(features, fg, bg):
        observed['fg'], observed['bg'] = fg, bg
        return original(features, fg, bg)
    model.channel_weights = record
    handle = model.head.register_forward_pre_hook(lambda _m, args: observed.update(head=args[0]))
    model(early, deep, logits)
    handle.remove()
    assert torch.count_nonzero(observed['bg']) == 0
    features = torch.cat((model.early(early), F.interpolate(model.deep(deep), size=(8, 8), mode='bilinear', align_corners=False)), dim=1)
    expected = F.cosine_similarity(features, observed['fg'], dim=1, eps=1e-6)
    torch.testing.assert_close(observed['head'][:, -2], expected)


def test_rgb_contrast_uses_rgb_map_and_validates_input():
    model = _model('rgb_contrast')
    early, deep, logits = _inputs()
    rgb = torch.rand(1, 3, 8, 8)
    head = []
    handle = model.head.register_forward_pre_hook(lambda _m, args: head.append(args[0]))
    model(early, deep, logits, rgb)
    handle.remove()
    prob = torch.sigmoid(logits)
    fg = (rgb * prob).sum((-2, -1), keepdim=True) / prob.sum((-2, -1), keepdim=True).clamp_min(1e-6)
    bg = (rgb * (1-prob)).sum((-2, -1), keepdim=True) / (1-prob).sum((-2, -1), keepdim=True).clamp_min(1e-6)
    expected = F.cosine_similarity(rgb, fg, dim=1, eps=1e-6) - F.cosine_similarity(rgb, bg, dim=1, eps=1e-6)
    torch.testing.assert_close(head[0][:, -2], expected)
    with pytest.raises(ValueError):
        model(early, deep, logits)
    with pytest.raises(ValueError):
        _model('reference')(early, deep, logits, rgb)


def test_rgb_loader_binds_image_hash_and_geometry(tmp_path):
    array = np.arange(6*8*3, dtype=np.uint8).reshape(6, 8, 3)
    image = tmp_path/'frame.png'
    Image.fromarray(array, 'RGB').save(image)
    item = {'image': str(image), 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
    actual = read_rgb(item, (6, 8))
    expected = np.asarray(Image.fromarray(array, 'RGB').resize((1024, 1024)))
    np.testing.assert_array_equal((actual[0].permute(1, 2, 0).numpy()*255).round().astype(np.uint8), expected)
    with pytest.raises(ValueError, match='geometry'):
        read_rgb(item, (8, 6))
    item['image_sha256'] = '0'*64
    with pytest.raises(ValueError, match='frozen manifest'):
        read_rgb(item, (6, 8))


def test_control_checkpoint_mismatch_and_degenerate_logits():
    model = _model('foreground_only')
    with pytest.raises(ValueError, match='Checkpoint control'):
        model.load_trained_state({'control': 'early_only'})
    early, deep, logits = _inputs()
    for value in (-1024, 1024):
        assert torch.isfinite(model(early, deep, torch.full_like(logits, value))).all()


@pytest.mark.parametrize('arm', ('reference', *ARMS))
def test_all_controls_backpropagate_finite_gradients(arm):
    model = _model(arm)
    early, deep, logits = _inputs()
    args = (early, deep, logits, torch.rand(1, 3, 8, 8)) if arm == 'rgb_contrast' else (early, deep, logits)
    model(*args).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
