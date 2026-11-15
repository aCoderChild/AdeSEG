"""Fail-closed tests for the frozen multiseed live-checkpoint adapter."""
import json

import pytest

from modeling import lean_boundary_multiseed_window_runtime_v1 as runtime


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def frozen_study(tmp_path, monkeypatch):
    seed = 20261125
    monkeypatch.setattr(runtime, 'SEED', seed)
    monkeypatch.setattr(runtime, 'EXPERIMENT_ROOT', tmp_path)
    monkeypatch.setenv('ADESEG_BOUNDARY_MULTISEED', str(seed))
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    study = tmp_path / str(seed)
    checkpoint = tmp_path / 'backbone.pt'
    checkpoint.write_bytes(b'actual backbone')
    manifest = {'checkpoint_sha256': runtime.digest(checkpoint)}
    manifest_sha = 'manifest'
    plan_path = study / 'plan/plan.json'
    parity_path = study / 'parity/parity.json'
    _write(plan_path, {'seed': seed})
    _write(parity_path, {'passed': True})
    monkeypatch.setattr(runtime, 'load_prepared', lambda _: (manifest, manifest_sha))
    monkeypatch.setattr(runtime, 'require_plan', lambda *_: {'seed': seed})
    monkeypatch.setattr(runtime, 'require_parity', lambda *_: None)
    monkeypatch.setattr(runtime, 'verify_live_identity', lambda *_: None)
    monkeypatch.setattr(runtime, 'dependencies', lambda: {'source': 'hash'})
    joint_path = tmp_path / 'joint_gate/joint.json'
    _write(joint_path, {'passed': True})
    _write(joint_path.parent / 'completion.json', {'passed': True, 'joint_sha256': runtime.digest(joint_path)})
    monkeypatch.setattr(runtime, 'check_joint_gate', lambda *_: {'passed': True, 'label_guard': {'open_attempts': 0}})
    runs = {}
    arm_evidence = {}
    for arm in runtime.ARMS:
        folder = study / 'runs' / arm
        for name in ('checkpoint.pt', 'completion.json', 'frozen_predictions.json'):
            _write(folder / name, {'arm': arm, 'name': name})
        runs[arm] = {'folder': folder, 'payload': {'arm': arm}}
        arm_evidence[arm] = {'checkpoint_sha256': runtime.digest(folder/'checkpoint.pt'),
                             'completion_sha256': runtime.digest(folder/'completion.json'),
                             'frozen_sha256': runtime.digest(folder/'frozen_predictions.json')}
    monkeypatch.setattr(runtime, 'check_frozen_arm', lambda folder, arm, *_: runs[arm])
    verification = study / 'verification/verification.json'
    _write(verification, {'passed': True, 'manifest_sha256': manifest_sha,
                          'plan_sha256': runtime.digest(plan_path), 'parity_sha256': runtime.digest(parity_path),
                          'dependency_sha256': {'source': 'hash'}, 'label_guard': {'open_attempts': 0},
                          'runs': arm_evidence})
    _write(verification.parent/'completion.json', {'passed': True, 'verification_sha256': runtime.digest(verification)})
    score = study/'scores/scores.json'
    _write(score, {'passed': True, 'manifest_sha256': manifest_sha, 'plan_sha256': runtime.digest(plan_path),
                   'verification_sha256': runtime.digest(verification), 'joint_gate_sha256': runtime.digest(joint_path)})
    audit = study/'independent_audit/audit.json'
    _write(audit, {'passed': True, 'seed': seed, 'study_path': str(study), 'error_count': 0,
                   'mask_files_checked': 600, 'actual_manifest_sha256': manifest_sha, 'recorded_manifest_sha256': manifest_sha,
                   'plan_sha256': runtime.digest(plan_path), 'parity_sha256': runtime.digest(parity_path),
                   'verification_sha256': runtime.digest(verification), 'scores_sha256': runtime.digest(score),
                   'joint_gate_sha256': runtime.digest(joint_path), 'arm_provenance': arm_evidence,
                   'audit_script_sha256': runtime.digest(runtime.ROOT/'scripts/audit_source_group_boundary_multiseed_v1.py')})
    return tmp_path, seed, checkpoint


def test_verified_study_selects_actual_seed_and_candidate(frozen_study):
    root, seed, checkpoint = frozen_study
    _, _, _, payload, evidence = runtime.verify_multiseed_boundary_study(root, seed, checkpoint)
    assert payload['arm'] == 'region_boundary'
    assert evidence['seed'] == seed
    assert evidence['candidate_checkpoint_sha256'] == runtime.digest(root/str(seed)/'runs/region_boundary/checkpoint.pt')


@pytest.mark.parametrize('field,value', [('seed', 20261135), ('error_count', 1), ('mask_files_checked', 599)])
def test_rejects_wrong_or_failed_independent_audit(frozen_study, field, value):
    root, seed, checkpoint = frozen_study
    path = root/str(seed)/'independent_audit/audit.json'
    data = json.loads(path.read_text()); data[field] = value; _write(path, data)
    with pytest.raises(ValueError):
        runtime.verify_multiseed_boundary_study(root, seed, checkpoint)


def test_rejects_replaced_candidate_checkpoint(frozen_study):
    root, seed, checkpoint = frozen_study
    (root/str(seed)/'runs/region_boundary/checkpoint.pt').write_bytes(b'wrong seed weights')
    with pytest.raises(ValueError, match='does not bind'):
        runtime.verify_multiseed_boundary_study(root, seed, checkpoint)


def test_rejects_seed_or_backbone_mismatch(frozen_study, monkeypatch):
    root, seed, checkpoint = frozen_study
    with pytest.raises(ValueError, match='seed environment'):
        runtime.verify_multiseed_boundary_study(root, 20261135, checkpoint)
    checkpoint.write_bytes(b'different backbone')
    with pytest.raises(ValueError, match='Backbone checkpoint'):
        runtime.verify_multiseed_boundary_study(root, seed, checkpoint)


def test_rejects_study_outside_requested_experiment_root(frozen_study, tmp_path, monkeypatch):
    root, seed, checkpoint = frozen_study
    monkeypatch.setattr(runtime, 'EXPERIMENT_ROOT', tmp_path/'other')
    with pytest.raises(ValueError, match='requested Drive experiment'):
        runtime.verify_multiseed_boundary_study(root, seed, checkpoint)
