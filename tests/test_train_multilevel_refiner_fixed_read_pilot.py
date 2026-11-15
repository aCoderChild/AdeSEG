import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from scripts import train_multilevel_refiner_fixed_read_pilot as pilot
from scripts.memory_token_io import digest


def test_native_single_stage_geometry_and_gradient():
    low = torch.randn(1, 1, 8, 8, requires_grad=True)
    result = pilot.to_video(low, (45, 67))
    expected = F.interpolate(low, size=(45, 67), mode='bilinear', align_corners=False)
    two_stage = F.interpolate(F.interpolate(low, size=(1024, 1024), mode='bilinear', align_corners=False), size=(45, 67), mode='bilinear', align_corners=False)
    assert not torch.allclose(expected, two_stage)
    torch.testing.assert_close(result, expected)
    target = torch.zeros_like(result); target[..., 4:7, 10:13] = 1
    pilot.loss_fn(result, target).backward()
    assert torch.isfinite(low.grad).all() and low.grad.abs().sum() > 0


def test_target_checks_shape_and_integrity(tmp_path):
    path = tmp_path / 'mask.png'
    Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).save(path)
    item = {'label': str(path), 'label_sha256': digest(path)}
    assert pilot.load_target(item, (2, 2)).sum() == 2
    with pytest.raises(ValueError, match='geometry'):
        pilot.load_target(item, (2, 3))
    item['label_sha256'] = 'bad'
    with pytest.raises(ValueError, match='changed'):
        pilot.load_target(item, (2, 2))


def test_manifest_split_excludes_priority_and_duplicates(tmp_path):
    samples = [{'split': split, 'sequence': seq, 'frame': n} for split, seqs in [('train', ('seq2', 'seq3', 'seq4')), ('validation', ('seq9', 'seq10'))] for seq in seqs for n in range(5)]
    manifest = {'passed': True, 'samples': samples, 'source_sha256': digest(pilot.ROOT / 'scripts/preflight_multilevel_refiner_pilot.py')}
    path = tmp_path / 'manifest.json'
    def write():
        path.write_text(json.dumps(manifest)); (tmp_path / 'completion.json').write_text(json.dumps({'passed': True, 'manifest_sha256': digest(path)}))
    write()
    assert len(pilot.check_manifest(tmp_path)[0]['samples']) == 25
    samples[0]['sequence'] = 'seq11'; write()
    with pytest.raises(ValueError, match='split'):
        pilot.check_manifest(tmp_path)


def test_mask_only_control_removes_features_without_altering_logits():
    early = torch.randn(1, 32, 4, 4); deep = torch.randn(1, 256, 2, 2)
    a, b = pilot.input_features(early, deep, 'mask_only')
    assert not a.any() and not b.any()
    a, b = pilot.input_features(early, deep, 'multilevel')
    assert a is early and b is deep
    with pytest.raises(ValueError, match='Unknown'):
        pilot.input_features(early, deep, 'other')
