import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.memory_token_io import digest
import scripts.run_verified_refiner_priority_development as runner


def write_json(path, value):
    path.write_text(json.dumps(value))


def make_preflight(tmp_path, image):
    folder = tmp_path / 'source' / 'seq11'
    folder.mkdir(parents=True)
    write_json(folder / 'prompt.json', {'first_detection_index': 0,
                                      'box': [1, 2, 3, 4], 'points': [[2, 3], [4, 5]]})
    write_json(folder / 'frames.json', [{'image_sha256': digest(image), 'present': True}])
    pre = tmp_path / 'preflight'
    pre.mkdir()
    write_json(pre / 'manifest.json', {'sequences': {'seq11': {
        'source': str(folder.parent), 'prompt_sha256': digest(folder / 'prompt.json'),
        'frames_metadata_sha256': digest(folder / 'frames.json')}}})
    write_json(pre / 'completion.json', {'passed': True, 'manifest_sha256': digest(pre / 'manifest.json')})
    return pre, folder


def test_priority_inputs_verify_prompt_and_actual_image_without_opening_mask(tmp_path, monkeypatch):
    image = tmp_path / 'image.jpg'
    image.write_bytes(b'source image')
    pre, folder = make_preflight(tmp_path, image)
    monkeypatch.setattr(runner, 'load_manifest', lambda *_: {'seq11': [SimpleNamespace(image_path=image)]})
    monkeypatch.setattr(runner, 'prepare_archived_prompt', lambda *_: (
        0, np.ones((2, 2), dtype=bool), (0, 0, 2, 2),
        {'prompt_metadata_sha256': digest(folder / 'prompt.json'), 'prompt_mask_sha256': 'mask'}))
    frames, prompt, _, view, mask, provenance = runner.priority_inputs(
        'seq11', pre, None, None, tmp_path / 'ckpt', tmp_path / 'detector')
    assert len(frames) == 1 and prompt['first_detection_index'] == 0
    assert view == {'kind': 'full_source_frame'}
    assert mask.all() and provenance['prompt_mask_sha256'] == 'mask'
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='images changed'):
        runner.priority_inputs('seq11', pre, None, None, tmp_path / 'ckpt', tmp_path / 'detector')
    image.write_bytes(b'source image')
    (folder / 'prompt.json').write_text('{}')
    with pytest.raises(ValueError, match='prompt'):
        runner.priority_inputs('seq11', pre, None, None, tmp_path / 'ckpt', tmp_path / 'detector')


def test_seq23_cannot_silently_use_uncropped_source(tmp_path, monkeypatch):
    image = tmp_path / 'image.jpg'
    image.write_bytes(b'frame')
    pre, _ = make_preflight(tmp_path, image)
    monkeypatch.setattr(runner, 'load_manifest', lambda *_: {'seq23': [SimpleNamespace(image_path=image)]})
    with pytest.raises(ValueError, match='cropped-prompt'):
        runner.priority_inputs('seq23', pre, None, None, tmp_path / 'ckpt', tmp_path / 'detector')


def test_cross_arm_runtime_checks_live_tensor_fingerprints_and_device(tmp_path):
    fields = ('candidate_checkpoint_sha256', 'control_checkpoint_sha256', 'selection_auditor_sha256',
              'underlying_screen_sha256',
              'baseline_reference_sha256', 'candidate_format', 'additional_controls',
              'model_tensor_count', 'refiner_tensor_count', 'source_sha256',
              'device', 'torch_version', 'python_version', 'platform', 'machine',
              'mps_device_count', 'gpu_inventory', 'gpu_inventory_note',
              'fallback', 'fast_math', 'seed', 'baseline_replays',
              'memory_budget', 'insurance_reserve')
    records = []
    for name in ('baseline', 'current'):
        folder = tmp_path / name
        folder.mkdir()
        write_json(folder / 'model_tensor_fingerprints.json', {'parameter:x': 'a'})
        write_json(folder / 'refiner_tensor_fingerprints.json', {'parameter:y': 'b'})
        data = {field: 'same' for field in fields}
        data.update({'passed': True,
                     'model_tensor_fingerprints_sha256': digest(folder / 'model_tensor_fingerprints.json'),
                     'refiner_tensor_fingerprints_sha256': digest(folder / 'refiner_tensor_fingerprints.json')})
        write_json(folder / 'runtime_verification.json', data)
        records.append((folder, data))
    runner.verify_cross_arm_runtime(records[0][0], records[1][0])
    records[1][1]['device'] = 'cpu'
    write_json(records[1][0] / 'runtime_verification.json', records[1][1])
    with pytest.raises(ValueError, match='runtime identity'):
        runner.verify_cross_arm_runtime(records[0][0], records[1][0])
    records[1][1]['device'] = 'same'
    write_json(records[1][0] / 'runtime_verification.json', records[1][1])
    # A self-consistent but different live refiner must still be rejected.
    write_json(records[1][0] / 'refiner_tensor_fingerprints.json', {'parameter:y': 'new'})
    records[1][1]['refiner_tensor_fingerprints_sha256'] = digest(records[1][0] / 'refiner_tensor_fingerprints.json')
    write_json(records[1][0] / 'runtime_verification.json', records[1][1])
    with pytest.raises(ValueError, match='runtime identity'):
        runner.verify_cross_arm_runtime(records[0][0], records[1][0])
    records[1][1]['refiner_tensor_fingerprints_sha256'] = records[0][1]['refiner_tensor_fingerprints_sha256']
    write_json(records[1][0] / 'runtime_verification.json', records[1][1])
    (records[1][0] / 'refiner_tensor_fingerprints.json').write_text('{}')
    with pytest.raises(ValueError, match='fingerprints changed'):
        runner.verify_cross_arm_runtime(records[0][0], records[1][0])
