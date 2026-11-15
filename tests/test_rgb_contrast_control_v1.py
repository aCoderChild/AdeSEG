import hashlib
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1
from modeling.foreground_conditioned_refiner import _source_sha256
import modeling.multiscale_channel_value_refiner_v1 as baseline
import modeling.rgb_contrast_control_v1 as rgb_module
from modeling.rgb_contrast_control_v1 import RGBContrastControlV1, rgb_contrast
from scripts.load_aligned_rgb_control_v1 import load_aligned_rgb


def test_rgb_replaces_only_contrast_head_channel_and_preserves_parameters():
    config = dict(early_channels=2, deep_channels=3, hidden=4, attention_size=3)
    reference = MultiscaleChannelValueRefinerV1(mode='foreground_background_asinh', **config)
    candidate = RGBContrastControlV1(**config)
    candidate.load_state_dict(reference.state_dict())
    assert [(k, tuple(v.shape)) for k, v in candidate.state_dict().items()] == [
        (k, tuple(v.shape)) for k, v in reference.state_dict().items()]
    e, d = torch.randn(1, 2, 4, 4), torch.randn(1, 3, 2, 2)
    logits, rgb = torch.randn(1, 1, 8, 8), torch.rand(1, 3, 8, 8)
    captured = []
    a = reference.head.register_forward_hook(lambda m, args, out: captured.append(args[0].clone()))
    b = candidate.head.register_forward_hook(lambda m, args, out: captured.append(args[0].clone()))
    assert torch.equal(reference(e, d, logits), candidate(e, d, logits, rgb))
    a.remove(); b.remove()
    assert torch.equal(captured[0][:, :-2], captured[1][:, :-2])
    assert torch.equal(captured[0][:, -1:], captured[1][:, -1:])
    assert torch.equal(captured[1][:, -2:-1], rgb_contrast(rgb, logits, 3))
    assert not torch.equal(captured[0][:, -2:-1], captured[1][:, -2:-1])
    assert len(candidate.head._forward_pre_hooks) == 0


def test_hook_is_removed_on_error_and_gradient_is_finite():
    model = RGBContrastControlV1(early_channels=2, deep_channels=3, hidden=4)
    e, d = torch.randn(1, 2, 4, 4), torch.randn(1, 3, 2, 2)
    logits, rgb = torch.randn(1, 1, 8, 8), torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError):
        model(e[:, :1], d, logits, rgb)
    assert len(model.head._forward_pre_hooks) == 0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(e, d, logits, rgb), torch.ones_like(logits))
    loss.backward()
    assert torch.isfinite(model.head[-1].bias.grad).all() and model.head[-1].bias.grad.abs().max() > 0
    with pytest.raises(ValueError, match='invalid'):
        rgb_contrast(rgb * 2, logits, 3)


def test_rgb_checkpoint_requires_both_exact_sources():
    config = dict(early_channels=2, deep_channels=3, hidden=4, attention_size=3)
    net = RGBContrastControlV1(**config)
    hashes = dict(backbone_sha256='a'*64, feature_pipeline_sha256='b'*64, train_manifest_sha256='c'*64)
    payload = dict(format='rgb_contrast_control_v1', research_only=True, training_steps=1,
                   config=config, state_dict=copy.deepcopy(net.state_dict()),
                   control_source_sha256=_source_sha256(Path(rgb_module.__file__)),
                   baseline_source_sha256=_source_sha256(Path(baseline.__file__)), **hashes)
    net.load_trained_state(payload, **hashes)
    with pytest.raises(ValueError, match='checkpoint or source'):
        net.load_trained_state(dict(payload, baseline_source_sha256='d'*64), **hashes)
    net.eval()
    with torch.no_grad(), pytest.raises(RuntimeError, match='Untrained'):
        net(torch.randn(1, 2, 4, 4), torch.randn(1, 3, 2, 2), torch.randn(1, 1, 8, 8), torch.rand(1, 3, 8, 8))
    assert len(net.head._forward_pre_hooks) == 0


def test_hash_bound_seq23_rgb_crop_and_index_validation(tmp_path):
    pixels = np.zeros((720, 1280, 3), np.uint8)
    pixels[:, :446] = [255, 0, 0]
    pixels[:, 446:] = [0, 128, 255]
    path = tmp_path / 'image.png'; Image.fromarray(pixels).save(path)
    frame = SimpleNamespace(image_path=path, frame_index=123, video_id='seq23')
    record = {'frame': 8, 'source_frame_index': 123, 'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    rgb = load_aligned_rgb(frame, record, 'seq23', 8, [446, 0, 1280, 720], (720, 834))
    assert rgb.shape == (1, 3, 720, 834)
    assert torch.equal(rgb[0, :, 0, 0], torch.tensor([0., 128/255, 1.]))
    with pytest.raises(ValueError, match='ordinal'):
        load_aligned_rgb(frame, record, 'seq23', 9, [446, 0, 1280, 720], (720, 834))
    with pytest.raises(ValueError, match='ordinal'):
        load_aligned_rgb(SimpleNamespace(image_path=path, frame_index=123, video_id='seq20'), record,
                         'seq23', 8, [446, 0, 1280, 720], (720, 834))
    with pytest.raises(ValueError, match='viewport'):
        load_aligned_rgb(frame, record, 'seq23', 8, [0, 0, 1280, 720], (720, 1280))
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes'):
        load_aligned_rgb(frame, record, 'seq23', 8, [446, 0, 1280, 720], (720, 834))
