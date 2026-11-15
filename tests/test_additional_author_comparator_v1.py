import copy
import json

import numpy as np
from PIL import Image
import pytest

from scripts import run_additional_author_comparator_v1 as run
from scripts import verify_additional_author_comparator_gate_v1 as verify
from scripts import run_additional_author_comparator_study_v1 as study
from scripts.memory_token_io import digest


def test_all_study_runs_are_frozen_before_verification_scoring_and_audit(tmp_path):
    phases, mapping = study.commands(tmp_path)
    names = [name for name, _ in phases]
    assert len(mapping) == 18 and all(len(v) == 3 for v in mapping.values())
    assert sum(name.startswith('short_') for name in names) == 54
    assert sum(name.startswith('gate_') for name in names) == 18
    assert sum(name.startswith('full_') for name in names) == 54
    assert names[-3:] == ['verify', 'score', 'audit']


def test_input_dispatch_and_unsupported_view(monkeypatch):
    frames = [object()] * 9
    prompt = {'first_detection_index': 2}
    mask = np.ones((2, 3), dtype=bool)
    provenance = {'roi_xyxy': [0, 0, 3, 2]}
    result = (frames, prompt, 'sha', {'kind': 'full_source_frame'}, mask, provenance)
    calls = []
    def extended(*args):
        calls.append('extended')
        return result
    def remaining(*args):
        calls.append('remaining')
        return result
    monkeypatch.setattr(run, 'extended_inputs', extended)
    monkeypatch.setattr(run, 'remaining_inputs', remaining)
    assert run.input_data('seq8', 'e', 'r', 'c', 'd') == (frames, 2, mask, provenance)
    run.input_data('seq6', 'e', 'r', 'c', 'd')
    assert calls == ['extended', 'remaining']
    with pytest.raises(ValueError, match='Sequence'):
        run.input_data('seq23', 'e', 'r', 'c', 'd')
    monkeypatch.setattr(run, 'extended_inputs', lambda *args: (*result[:3], {'kind': 'other'}, *result[4:]))
    with pytest.raises(ValueError, match='view'):
        run.input_data('seq8', 'e', 'r', 'c', 'd')


def make_short_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'identity', lambda method, seq, c, d, ep, rp, device:
                        {'method': method, 'sequence': seq, 'device': device})
    dirs = []
    for i, device in enumerate(('cpu', 'mps', 'mps')):
        folder = tmp_path / str(i)
        folder.mkdir()
        records = []
        for frame in range(7):
            path = folder / f'{frame:06d}.png'
            Image.fromarray(np.zeros((5, 6), np.uint8)).save(path)
            records.append({'frame': frame, 'source_frame_index': frame, 'image_sha256': f'image{frame}',
                            'mask_sha256': digest(path), 'post_prompt': frame >= 2})
        receipt = {'passed': True, 'full_video': False, 'deterministic': True,
                   'identity': {'method': 'native', 'sequence': 'seq8', 'device': device},
                   'label_guard': {'open_attempts': 0}, 'method': 'native', 'sequence': 'seq8',
                   'prompt_frame': 2, 'prompt': {'p': 1}, 'state_sha256': 'state', 'noncausal': False,
                   'process_id': i + 1, 'run_nonce': f'nonce{i}', 'records': records}
        (folder / 'receipt.json').write_text(json.dumps(receipt))
        dirs.append(folder)
    return dirs


def test_short_gate_checks_distinct_processes_hashes_and_prompt(tmp_path, monkeypatch):
    dirs = make_short_runs(tmp_path, monkeypatch)
    gate = verify.audit(*dirs, 'c', 'd', 'e', 'r')
    assert gate['passed'] and len(gate['rows']) == 7
    assert gate['identity']['device'] == 'mps'
    with pytest.raises(ValueError, match='distinct'):
        verify.audit(dirs[0], dirs[1], dirs[1], 'c', 'd', 'e', 'r')
    receipt_path = dirs[2] / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    altered = copy.deepcopy(receipt)
    altered['prompt'] = {'p': 2}
    receipt_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='identity'):
        verify.audit(*dirs, 'c', 'd', 'e', 'r')
    receipt_path.write_text(json.dumps(receipt))
    Image.fromarray(np.ones((5, 6), np.uint8)).save(dirs[1] / '000003.png')
    with pytest.raises(ValueError, match='hashes'):
        verify.audit(*dirs, 'c', 'd', 'e', 'r')


def test_gate_reconstructs_and_rejects_tampering(tmp_path, monkeypatch):
    dirs = make_short_runs(tmp_path, monkeypatch)
    monkeypatch.setattr(run, 'EXPERIMENT_ROOT', tmp_path)
    gate = verify.audit(*dirs, 'c', 'd', 'e', 'r')
    gate['label_guard'] = {'open_attempts': 0}
    file = tmp_path / 'gate.json'
    file.write_text(json.dumps(gate))
    expected = {'method': 'native', 'sequence': 'seq8', 'device': 'mps'}
    args = ('c', 'd', 'e', 'r')
    assert run.verify_gate(file, expected, {'p': 1}, *args)['passed']
    with pytest.raises(ValueError, match='matching|Matching'):
        run.verify_gate(file, dict(expected, method='samurai'), {'p': 1}, *args)
    with pytest.raises(ValueError, match='matching|Matching'):
        run.verify_gate(file, expected, {'p': 3}, *args)
    altered = dict(gate, rows=[])
    file.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='Reconstructed'):
        run.verify_gate(file, expected, {'p': 1}, *args)
    file.write_text(json.dumps(gate))
    (dirs[1] / 'receipt.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='changed'):
        run.verify_gate(file, expected, {'p': 1}, *args)
