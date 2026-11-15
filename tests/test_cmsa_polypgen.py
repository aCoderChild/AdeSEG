from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_cmsa_polypgen import clips


def test_author_clip_order_and_full_first_save_coverage():
    frames = [SimpleNamespace(image_path=Path(f'/images/{i}.jpg'), mask_path=Path(f'/masks/{i}.png')) for i in range(7)]
    actual = clips(frames)
    assert [[Path(image).stem for image, _ in clip] for clip in actual] == [
        ['0', '1', '0', '1', '2', '3'],
        ['0', '1', '1', '2', '3', '4'],
        ['0', '1', '2', '3', '4', '5'],
        ['0', '1', '3', '4', '5', '6'],
    ]
    assert {image for clip in actual for image, _ in clip[2:]} == {str(f.image_path) for f in frames}
    with pytest.raises(ValueError):
        clips(frames[:3])


def test_adapter_matches_upstream_dataset(tmp_path, monkeypatch):
    import sys
    import numpy as np
    from PIL import Image
    import torch
    from scripts.run_cmsa_polypgen import author_dataset, load_author

    load_author()
    from lib.dataloader.dataloader import CMSAVideoDataset

    root = tmp_path / 'ClinicDB'
    images = root / 'set' / 'Frame' / 'seq3'
    masks = root / 'set' / 'GT' / 'seq3'
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    frames = []
    for i in range(5):
        image, mask = images / f'{i}.jpg', masks / f'{i}.png'
        Image.fromarray(np.full((12, 16, 3), i * 30, dtype=np.uint8)).save(image)
        Image.fromarray(np.full((12, 16), i % 2 * 255, dtype=np.uint8)).save(mask)
        frames.append(SimpleNamespace(image_path=image, mask_path=mask))
    adapted = author_dataset({'seq3': frames}, normalization='released')
    config = sys.modules['scripts.config'].config
    monkeypatch.setattr(config, 'dataset_root', str(root))
    monkeypatch.setattr(config, 'video_time_clips', 4)
    upstream = CMSAVideoDataset('set', transform=adapted.img_label_transform, ttype='eval')
    assert len(adapted) == len(upstream) == 2
    for i in range(len(upstream)):
        images_a, labels_a, paths_a = adapted[i]
        images_b, labels_b, paths_b = upstream[i]
        assert paths_a == paths_b
        assert torch.equal(images_a, images_b)
        assert torch.equal(labels_a, labels_b)


def test_corrected_normalization_uses_rgb_channels_not_columns():
    import torch
    from scripts.run_cmsa_polypgen import ChannelNormalizeVideo

    image = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) / 100
    original = image.clone()
    mean, std = [0.1, 0.2, 0.3], [0.5, 0.25, 0.125]
    labels = [object()]
    transformed, returned_labels = ChannelNormalizeVideo(mean, std)([image], labels)
    expected = (original - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    assert torch.equal(transformed[0], expected)
    assert torch.equal(image, original)
    assert returned_labels is labels


def test_author_attention_rejects_future_clip_tokens():
    import torch
    from scripts.run_cmsa_polypgen import load_author

    author = load_author()
    torch.manual_seed(42)
    layer = author.CMSANet().x2_ms_tca.eval()
    features = [torch.randn(6, 32, 2, 2) for _ in range(3)]
    changed = [x.clone() for x in features]
    for x in changed:
        x[4:] += 20
    with torch.no_grad():
        baseline = layer(*features, B=1, T=6)
        perturbed = layer(*changed, B=1, T=6)
    assert torch.equal(baseline[:4], perturbed[:4])
    assert not torch.equal(baseline[4:], perturbed[4:])


def test_normalization_comparison_pairs_and_refuses_mismatch(tmp_path):
    import json
    import subprocess
    import sys

    released, channel = tmp_path / 'released', tmp_path / 'channel'
    for path in (released, channel):
        path.mkdir()
    setup = {'author_revision': 'abc', 'checkpoint_sha256': '123', 'manifest_sha256': '456',
             'sequences': ['seq3'], 'clip_length': 4, 'probability_threshold': 128, 'device': 'cpu'}
    def write(path, name, value):
        (path / name).write_text(json.dumps(value))
    for path, mode, score in ((released, 'released', 0.4), (channel, 'channel', 0.6)):
        write(path, 'setup.json', {**setup, 'normalization': mode})
        write(path, 'results.json', {'equal_sequence_mean_present_dice': score,
              'per_sequence': {'seq3': {'mean_present_dice': score, 'prompt_index': 0,
                                'matched_targets': 1, 'present_targets': 1,
                                'gt_empty_targets': 0, 'false_alarms': 0}}})
        write(path, 'frames.json', [{'sequence': 'seq3', 'frame': 0, 'present': True}])
        write(path, 'matched_native_audit.json', {'status': 'passed', 'native_results_sha256': 'native',
              'matched_targets': 1, 'empty_targets': 0, 'cmsa_false_alarms': 0})
    script = Path(__file__).resolve().parents[1] / 'scripts/compare_cmsa_normalization.py'
    cmd = [sys.executable, str(script), '--released', str(released), '--channel', str(channel)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    report = json.loads((channel / 'normalization_comparison.json').read_text())
    assert report['mean_difference'] == pytest.approx(0.2)
    (channel / 'normalization_comparison.json').unlink()
    write(channel, 'setup.json', {**setup, 'normalization': 'channel', 'checkpoint_sha256': 'other'})
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'checkpoint_sha256' in result.stderr
