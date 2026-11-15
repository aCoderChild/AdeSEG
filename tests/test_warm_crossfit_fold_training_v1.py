"""Synthetic CPU unit tests; production workers use MPS and real provenance."""
import json

import pytest
import torch
from scripts.memory_token_io import digest

from scripts import warm_crossfit_fold_training_v1 as worker
from scripts.source_group_refiner_shared_v2 import initial_hash


class TinyRefiner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25]))

    def forward(self, early, deep, logits):
        return logits + self.weight


class FakeCache:
    def __init__(self, limit, loader): self.limit = limit
    def get(self, item, device): return worker.inputs(item, device)
    def verify_boundary(self, *, final=False): pass
    def stats(self): return {'limit_bytes': self.limit}


def test_fold_worker_trains_only_authorized_scheduled_sample(tmp_path, monkeypatch):
    item = {'sequence': 'seq8', 'frame': 0, 'origin': 'historical', 'label_sha256': 'sha'}
    monkeypatch.setattr(worker, 'samples_for', lambda _: ({'source_frames': {}}, {('seq8', 0): item}))
    monkeypatch.setattr(worker, 'fold_definition', lambda *a: ({'seq8'}, set(), {'seq6'}, [[['seq8', 0]]]))
    monkeypatch.setattr(worker, 'training_label_hashes', lambda *a: {})
    monkeypatch.setattr(worker, 'model_for', lambda _: TinyRefiner())
    monkeypatch.setattr(worker, 'VerifiedCaptureCache', FakeCache)
    observed = []
    monkeypatch.setattr(worker, 'inputs', lambda item, device: ((torch.zeros(1), torch.zeros(1),
                                                                 torch.zeros(1, 1, 2, 2)), (2, 2)))
    def target(item, shape, sha):
        observed.append((item['sequence'], sha))
        return torch.ones(1, 1, 2, 2)
    monkeypatch.setattr(worker, 'checked_train_target', target)
    monkeypatch.setattr(worker, 'cmsa_boundary_loss', lambda prediction, target: ((prediction-target)**2).mean())
    monkeypatch.setattr(worker, 'save_payload', lambda model, **kw: {'weight': model.weight.detach().cpu()})
    plan = {'initial_state_sha256': {'mask_only': initial_hash(TinyRefiner())}}
    out = tmp_path / 'run'; guard = {'blocked_open_attempts': 0}
    worker.train(plan, 'preflight', 1, 'mask_only', out, guard, device='cpu')
    receipt = json.loads((out / 'completion.json').read_text())
    assert observed == [('seq8', 'sha')]
    assert receipt['passed'] and receipt['training_steps'] == 1
    assert receipt['history'][0]['peak_gradient_l2'] > 0
    assert torch.load(out / 'checkpoint.pt', weights_only=True)['weight'] != .25


def test_nontraining_patient_stops_before_decoding(tmp_path, monkeypatch):
    item = {'sequence': 'seq6', 'frame': 0, 'origin': 'historical', 'label_sha256': 'sha'}
    monkeypatch.setattr(worker, 'samples_for', lambda _: ({'source_frames': {}}, {('seq6', 0): item}))
    monkeypatch.setattr(worker, 'fold_definition', lambda *a: ({'seq8'}, set(), {'seq6'}, [[['seq6', 0]]]))
    monkeypatch.setattr(worker, 'training_label_hashes', lambda *a: {})
    monkeypatch.setattr(worker, 'model_for', lambda _: TinyRefiner())
    monkeypatch.setattr(worker, 'VerifiedCaptureCache', FakeCache)
    monkeypatch.setattr(worker, 'inputs', lambda *a: pytest.fail('protected sample was read'))
    plan = {'initial_state_sha256': {'mask_only': initial_hash(TinyRefiner())}}
    with pytest.raises(ValueError, match='Non-training patient'):
        worker.train(plan, 'preflight', 1, 'mask_only', tmp_path/'run', {'blocked_open_attempts': 0}, device='cpu')


def test_checkpoint_for_wrong_fold_rejected_before_model_load():
    payload = {'crossfit_fold': 2, 'crossfit_arm': 'global', 'preflight_sha256': 'p',
               'aggregate_sha256': 'a'}
    with pytest.raises(ValueError, match='provenance'):
        worker.trained_model(payload, arm='global', fold=1, preflight={'aggregate_sha256': 'a'}, preflight_sha='p')


def test_label_free_prediction_and_phase_bound_training_tamper_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'BASE', tmp_path)
    item = {'sequence': 'seq8', 'frame': 0, 'origin': 'historical'}
    index = {'source_frames': {'seq8': [{'source_frame_index': 7}]}}
    monkeypatch.setattr(worker, 'samples_for', lambda _: (index, {('seq8', 0): item}))
    monkeypatch.setattr(worker, 'fold_definition', lambda *a: (set(), set(), {'seq8'}, [[['seq2', 1]]]))
    monkeypatch.setattr(worker, 'trained_model', lambda *a, **k: TinyRefiner())
    monkeypatch.setattr(worker, 'inputs', lambda item, device: ((torch.zeros(1), torch.zeros(1),
        torch.zeros(1, 1, 2, 2)), (2, 2)))
    training = tmp_path / 'train/fold1/mask_only'; training.mkdir(parents=True)
    ck = training / 'checkpoint.pt'; torch.save({'training_steps': 1}, ck)
    (training / 'completion.json').write_text(json.dumps({'passed': True, 'fold': 1, 'arm': 'mask_only',
        'preflight_sha256': 'p', 'checkpoint_sha256': digest(ck), 'training_steps': 1,
        'device': 'cpu', 'environment': None, 'label_guard': {'blocked_open_attempts': 0}}))
    phases = tmp_path/'phases.jsonl'
    def append_phase(name, artifact):
        log = tmp_path/(name + '.log'); log.write_text('done')
        with phases.open('a') as stream:
            stream.write(json.dumps({'phase': name, 'passed': True, 'artifact_path': str(artifact),
                'artifact_sha256': digest(artifact), 'log_sha256': digest(log)}) + '\n')
    append_phase('train_f1_mask_only', training/'completion.json')
    guard = {'open_attempts': 0}; prediction = tmp_path / 'predict/fold1/mask_only'
    worker.predict({}, 'p', 1, 'mask_only', training, prediction, guard, device='cpu')
    receipt = json.loads((prediction/'completion.json').read_text())
    assert receipt['rows'][0]['source_frame_index'] == 7
    assert (prediction/'seq8_000000_refined.png').exists() and guard['open_attempts'] == 0
    (training/'completion.json').write_text('{}')
    with pytest.raises(ValueError, match='phase artifact changed'):
        worker.predict({}, 'p', 1, 'mask_only', training, tmp_path/'different', guard, device='cpu')


def test_changed_warm_certificate_rejected_before_dataset_loading(tmp_path, monkeypatch):
    old = tmp_path/'old'; old.mkdir(); manifest = old/'manifest.json'; manifest.write_text('{}')
    aggregate_dir = tmp_path/'aggregate'; aggregate_dir.mkdir()
    aggregate_path = aggregate_dir/'captures.json'
    aggregate_path.write_text(json.dumps({'certificate_path': '/certificate', 'certificate_sha256': 'expected',
        'warm_gate_sha256': 'gate', 'warm_gate_path': '/gate',
        'prompt_policy': 'warm_box_points_then_commit_archived_mask'}))
    monkeypatch.setattr(worker, 'EXISTING', old)
    monkeypatch.setattr(worker, 'validate_plan', lambda: {'checkpoint_sha256': 'backbone',
        'existing_manifest_sha256': digest(manifest)})
    monkeypatch.setattr(worker, 'verify_certificate', lambda _: ({}, 'changed'))
    monkeypatch.setattr(worker, 'load_manifest', lambda *a: pytest.fail('dataset loaded before certificate check'))
    preflight = {'backbone_sha256': 'backbone', 'old_manifest_sha256': digest(manifest),
        'aggregate_path': str(aggregate_dir), 'aggregate_sha256': digest(aggregate_path),
        'certificate_sha256': 'expected', 'warm_gate_sha256': 'gate'}
    with pytest.raises(ValueError, match='certificate or gate changed'):
        worker.samples_for(preflight)
