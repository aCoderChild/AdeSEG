import json
import shutil

import pytest
import torch

from scripts import audit_boundary_mps_repeatability_v1 as probe
from scripts.memory_token_io import digest


def test_fingerprint_binds_values_shapes_and_types_but_not_dict_order():
    a = {'a': torch.tensor([[1.0, 2.0]]), 'b': [2, 3.0]}
    assert probe.fingerprint(a) == probe.fingerprint({'b': [2, 3.0], 'a': a['a'].clone()})
    assert probe.fingerprint(a) != probe.fingerprint({'a': torch.tensor([1.0, 2.0]), 'b': [2, 3.0]})
    assert probe.fingerprint(a) != probe.fingerprint({'a': torch.tensor([[1.0, 2.1]]), 'b': [2, 3.0]})
    with pytest.raises(ValueError, match='Nonfinite'):
        probe.fingerprint(float('nan'))


def test_numeric_difference_rejects_mismatched_structure():
    assert probe.max_difference({'x': torch.tensor([1., 2.])}, {'x': torch.tensor([1., 2.5])}) == .5
    with pytest.raises(ValueError, match='keys'):
        probe.max_difference({'x': 1.}, {'y': 1.})
    with pytest.raises(ValueError, match='metadata'):
        probe.max_difference(torch.ones(2), torch.ones(3))


def test_probe_selection_rejects_validation_or_duplicate_reads(monkeypatch):
    monkeypatch.setattr(probe, 'KEYS', (('seq2', 4), ('seq4', 24)))
    rows = [{'sequence': 'seq2', 'frame': 4, 'split': 'train'},
            {'sequence': 'seq4', 'frame': 24, 'split': 'train'}]
    assert probe.selected_rows({'samples': rows}) == rows
    with pytest.raises(ValueError, match='missing or duplicated'):
        probe.selected_rows({'samples': [rows[0], {**rows[1], 'split': 'validation'}]})
    with pytest.raises(ValueError, match='missing or duplicated'):
        probe.selected_rows({'samples': rows + [rows[0]]})


def make_run(folder, plan_path, plan, keys, *, change=False, process_id=1):
    folder.mkdir()
    steps = []
    for i, (seq, frame) in enumerate(keys):
        value = float(i + (1 if change and i == 1 else 0))
        snapshot = {'pre_model': {'w': torch.tensor([float(i)])},
                    'prediction': torch.tensor([value]), 'loss': value,
                    'gradients': {'w': torch.tensor([value])},
                    'post_model': {'w': torch.tensor([value])},
                    'optimizer': {'state': {0: {'step': torch.tensor(float(i + 1))}}}}
        file = folder/f'step_{i:02d}.pt'; torch.save(snapshot, file)
        steps.append({'index': i, 'sequence': seq, 'frame': frame,
                      'input_fingerprints': {'early': 'same', 'deep': 'same', 'logits': 'same', 'target': 'same'},
                      'snapshot_sha256': digest(file),
                      'fingerprints': {k: probe.fingerprint(snapshot[k]) for k in probe.STAGES}})
    receipt = {'passed': True, 'plan_sha256': digest(plan_path), 'source_sha256': plan['source_sha256'],
               'pythonhashseed': '0', 'environment': plan['environment'],
               'initial_state_sha256': plan['initial_state_sha256'],
               'process_id': process_id, 'process_token': str(process_id).zfill(32),
               'label_guard': {'blocked_open_attempts': 0}, 'steps': steps}
    (folder/'receipt.json').write_text(json.dumps(receipt))


def test_compare_identifies_first_differing_stage_and_rejects_tamper(tmp_path, monkeypatch):
    keys = (('seq2', 4), ('seq4', 24)); monkeypatch.setattr(probe, 'KEYS', keys)
    plan = {'scope': 'test', 'source_sha256': 'source', 'environment': {'device': 'test'},
            'initial_state_sha256': 'initial'}
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan))
    a, b = tmp_path/'a', tmp_path/'b'
    make_run(a, plan_path, plan, keys); make_run(b, plan_path, plan, keys, change=True, process_id=2)
    result = probe.compare(plan, plan_path, a, b)
    assert not result['all_stages_exact']
    assert result['first_difference'] == {'step': 1, 'stage': 'prediction', 'sequence': 'seq4', 'frame': 24}
    assert result['rows'][1]['stages']['prediction']['max_abs_difference'] == 1
    with pytest.raises(ValueError, match='distinct'):
        probe.compare(plan, plan_path, a, a)
    copied = tmp_path/'copied'; shutil.copytree(a, copied)
    with pytest.raises(ValueError, match='process identities'):
        probe.compare(plan, plan_path, a, copied)
    receipt = json.loads((b/'receipt.json').read_text()); receipt['process_id'] = 1
    (b/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='process identities'):
        probe.compare(plan, plan_path, a, b)
    receipt['process_id'] = 2; receipt['steps'][0]['input_fingerprints']['target'] = 'changed'
    (b/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='inputs differ'):
        probe.compare(plan, plan_path, a, b)
    receipt['steps'][0]['input_fingerprints']['target'] = 'same'
    (b/'receipt.json').write_text(json.dumps(receipt))
    (b/'step_00.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        probe.compare(plan, plan_path, a, b)
