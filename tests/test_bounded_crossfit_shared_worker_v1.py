"""Synthetic evidence and fail-closed adapter tests; no GPU or target data."""
import copy
import fcntl
import json
from pathlib import Path

import pytest
import torch

from scripts import bounded_crossfit_shared_worker_v1 as worker
from scripts import bounded_crossfit_supervision_v1 as supervision
from scripts import audit_bounded_crossfit_shared_probe_v1 as probe_auditor
from scripts import probe_bounded_crossfit_shared_v1 as probe_runner
from scripts import run_bounded_crossfit_shared_study_v1 as supervisor
from scripts import verify_bounded_crossfit_registry_v1 as verifier
from scripts.memory_token_io import digest


def test_training_worker_is_disabled_before_any_input_access(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ENABLE_REVIEWED_TRAINING', False)
    with pytest.raises(ValueError, match='awaits reviewed MPS probe'):
        worker.train_fold({}, 1, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_probe_runner_is_disabled_before_reading_any_data(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_runner, 'ENABLE_REVIEWED_PROBE', False)
    monkeypatch.setattr(probe_runner, 'BASE', tmp_path)
    monkeypatch.setattr('sys.argv', ['probe', '--registry', str(tmp_path / 'missing'),
                                   '--verification', str(tmp_path / 'missing-proof'),
                                   '--batch-root', str(tmp_path),
                                   '--output', str(tmp_path / 'output')])
    with pytest.raises(ValueError, match='awaits independent review'):
        probe_runner.main()
    with pytest.raises(ValueError, match='awaits independent review'):
        probe_runner.probe({}, '', '', tmp_path / 'output', {'open_attempts': 0}, batch_root=tmp_path)
    assert not (tmp_path / 'output').exists()


def test_verifier_rejects_changed_registry_even_with_rewritten_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, 'BASE', tmp_path)
    monkeypatch.setattr(verifier, 'sources', lambda: {'v': 'a' * 64})
    original = {'passed': True, 'training_authorized': False, 'format': 'bounded_crossfit_registry_v1',
                'preflight_path': '/preflight', 'preflight_sha256': 'pre', 'preflight_completion_sha256': 'done',
                'handoff_sha256': 'handoff', 'seq6_proof_sha256': 's6', 'seq11_proof_sha256': 's11'}
    monkeypatch.setattr(verifier.registry, 'build', lambda _: copy.deepcopy(original))
    folder = tmp_path / 'registry'; folder.mkdir()
    def save(value):
        path = folder / 'registry.json'
        path.write_text(json.dumps(value | {'label_guard': {'open_attempts': 0}}))
        (folder / 'completion.json').write_text(json.dumps({'passed': True, 'training_authorized': False,
            'registry_sha256': digest(path), 'label_guard': {'open_attempts': 0}}))
    save(original)
    assert verifier.verify(folder)['training_authorized'] is False
    save(original | {'handoff_sha256': 'changed'})
    with pytest.raises(ValueError, match='reconstruction'):
        verifier.verify(folder)


def test_probe_check_requires_audited_all_arms_and_untampered_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'BASE', tmp_path)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    scripts = tmp_path / 'scripts'; scripts.mkdir()
    runner = scripts / 'probe_bounded_crossfit_shared_v1.py'; runner.write_text('probe')
    auditor = scripts / 'audit_bounded_crossfit_shared_probe_v1.py'; auditor.write_text('audit')
    folder = tmp_path / 'probe'; folder.mkdir()
    identity = {'source': 's' * 64}; registry_sha = 'r' * 64; verification_sha = 'v' * 64
    initial = {arm: 'i' * 64 for arm in worker.ARMS}
    registry = {'samples': [{'sequence': 'seq6', 'frame': 2}, {'sequence': 'seq23', 'frame': 3}],
                'initial_state_sha256': initial, 'environment': {'device': 'mps'}}
    checks = []
    for sequence, frame in (('seq6', 2), ('seq23', 3)):
        for arm in worker.ARMS:
            path = folder / f'{sequence}_{arm}.pt'; path.write_bytes((sequence + arm).encode())
            checks.append({'sequence': sequence, 'frame': frame, 'arm': arm, 'optimizer_updated': True,
                           'gradient_l2': 1.0, 'head_bias_gradient_abs': 0.1, 'initial_state_sha256': initial[arm],
                           'video_shape': [720, 834] if sequence == 'seq23' else [720, 1280], 'synthetic_loss': 1.0,
                           'checkpoint_path': str(path), 'checkpoint_sha256': digest(path)})
    probe_path = folder / 'completion.json'
    record = {'passed': True, 'format': 'bounded_shared_six_arm_probe_v1', 'device': 'mps',
              'registry_sha256': registry_sha, 'verification_sha256': verification_sha, 'preflight_sha256': 'pre',
              'source_sha256': identity, 'runner_sha256': digest(runner), 'environment': registry['environment'],
              'label_guard': {'open_attempts': 0}, 'checks': checks,
              'peak_process_rss_bytes': 100, 'max_process_rss_bytes': 200}
    audit_path = tmp_path / 'audit.json'
    def save():
        probe_path.write_text(json.dumps(record))
        audit_path.write_text(json.dumps({'passed': True, 'format': 'bounded_shared_six_arm_probe_audit_v1',
            'probe_sha256': digest(probe_path), 'registry_sha256': registry_sha,
            'verification_sha256': verification_sha, 'source_sha256': identity, 'auditor_sha256': digest(auditor),
            'peak_process_rss_bytes': record.get('peak_process_rss_bytes'),
            'max_process_rss_bytes': record.get('max_process_rss_bytes'),
            'error_count': 0, 'audited_checks': len(checks), 'label_guard': {'open_attempts': 0}}))
    save()
    assert worker.check_probe(probe_path, audit_path, registry, registry_sha, verification_sha, 'pre', identity)[0] == digest(probe_path)
    record['checks'][0]['optimizer_updated'] = False; save()
    with pytest.raises(ValueError, match='update evidence'):
        worker.check_probe(probe_path, audit_path, registry, registry_sha, verification_sha, 'pre', identity)
    record['checks'][0]['optimizer_updated'] = True; save()
    Path(record['checks'][0]['checkpoint_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        worker.check_probe(probe_path, audit_path, registry, registry_sha, verification_sha, 'pre', identity)


@pytest.mark.parametrize('peak, limit', [(None, 200), (201, 200), (100, None), (0, 200)])
def test_probe_rejects_missing_or_excess_recorded_rss(tmp_path, monkeypatch, peak, limit):
    monkeypatch.setattr(worker, 'BASE', tmp_path); monkeypatch.setattr(worker, 'ROOT', tmp_path)
    scripts = tmp_path / 'scripts'; scripts.mkdir()
    runner = scripts / 'probe_bounded_crossfit_shared_v1.py'; runner.write_text('probe')
    auditor = scripts / 'audit_bounded_crossfit_shared_probe_v1.py'; auditor.write_text('audit')
    probe = tmp_path / 'probe.json'; audit = tmp_path / 'audit.json'
    r = {'passed': True, 'format': 'bounded_shared_six_arm_probe_v1', 'device': 'mps',
         'registry_sha256': 'r', 'verification_sha256': 'v', 'preflight_sha256': 'p',
         'source_sha256': {}, 'runner_sha256': digest(runner), 'environment': {},
         'label_guard': {'open_attempts': 0}, 'checks': [],
         'peak_process_rss_bytes': peak, 'max_process_rss_bytes': limit}
    probe.write_text(json.dumps(r))
    audit.write_text(json.dumps({'passed': True, 'format': 'bounded_shared_six_arm_probe_audit_v1',
        'probe_sha256': digest(probe), 'registry_sha256': 'r', 'verification_sha256': 'v',
        'source_sha256': {}, 'auditor_sha256': digest(auditor), 'peak_process_rss_bytes': peak,
        'max_process_rss_bytes': limit, 'error_count': 0, 'audited_checks': 12,
        'label_guard': {'open_attempts': 0}}))
    with pytest.raises(ValueError, match='probe or its independent audit'):
        worker.check_probe(probe, audit, {'environment': {}, 'samples': [], 'initial_state_sha256': {}},
                           'r', 'v', 'p', {})


def test_supervisor_queues_only_declared_training_phases_and_checks_disk_first(tmp_path, monkeypatch):
    plan = supervisor.phases(tmp_path, tmp_path / 'registry', 512)
    names = [name for name, _, _ in plan]
    assert names[:9] == ['verification', 'probe', 'probe_audit', 'authorization',
                         'train_f1', 'train_f2', 'train_f3', 'train_f4', 'train_f5']
    assert len([name for name in names if name.startswith('predict_')]) == 30
    assert names[-3:] == ['verify', 'score', 'audit']
    monkeypatch.setattr(supervisor.shutil, 'disk_usage', lambda _: type('D', (), {'free': 1})())
    with pytest.raises(ValueError, match='Resource guard'):
        supervisor.run_monitored(['should-not-run'], tmp_path / 'phase.log', tmp_path,
                                 min_disk=2, min_ram=0, max_rss=1)
    assert not (tmp_path / 'phase.log').exists()


def test_supervision_rejects_standalone_parent_before_probe_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision, 'BASE', tmp_path); monkeypatch.setattr(supervision, 'OUT', tmp_path)
    monkeypatch.setattr(worker, 'sources', lambda: {})
    registry = tmp_path / 'registry'; registry.mkdir()
    (registry / 'registry.json').write_text('{}'); (registry / 'completion.json').write_text('{}')
    root = tmp_path / 'study'; root.mkdir()
    manifest = {'format': 'bounded_shared_crossfit_training_study_v1', 'registry_path': str(registry),
                'registry_sha256': digest(registry / 'registry.json'),
                'registry_completion_sha256': digest(registry / 'completion.json'), 'source_sha256': {},
                'cache_mib': 0, 'min_disk_bytes': 26 * 1024**3, 'min_available_ram_bytes': 4 * 1024**3,
                'max_process_rss_bytes': 6 * 1024**3, 'supervisor_pid': -1, 'supervisor_create_time': 0.0}
    (root / 'run_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Supervisor parent differs'):
        supervision.assert_supervised(root, 'probe', root / 'probe')
    monkeypatch.setattr(probe_runner, 'BASE', tmp_path)
    monkeypatch.setattr(probe_runner, 'ENABLE_REVIEWED_PROBE', True)
    monkeypatch.setattr('sys.argv', ['probe', '--registry', str(registry), '--verification', str(tmp_path / 'proof'),
                                   '--batch-root', str(root), '--output', str(root / 'probe')])
    monkeypatch.setattr(probe_runner.torch.backends.mps, 'is_available', lambda: pytest.fail('MPS was reached'))
    with pytest.raises(ValueError, match='Supervisor parent differs'):
        probe_runner.main()
    monkeypatch.setattr(worker, 'ENABLE_REVIEWED_TRAINING', True)
    monkeypatch.setattr(worker, 'install_fold_training_guard', lambda *args: pytest.fail('Label guard was reached'))
    with pytest.raises(ValueError, match='Supervisor parent differs'):
        worker.train_fold({'training_authorized': True}, 1, root / 'fold1', batch_root=root, cache_mib=0)
    assert not (root / 'probe').exists()


def test_supervision_compares_actual_parent_and_child_python_executables(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision, 'BASE', tmp_path); monkeypatch.setattr(supervision, 'OUT', tmp_path)
    monkeypatch.setattr(worker, 'sources', lambda: {})
    current_executable = supervision.psutil.Process().exe()
    class Parent:
        def is_running(self): return True
        def create_time(self): return 123.5
        def cmdline(self): return [current_executable, str(supervision.ROOT / 'scripts/run_bounded_crossfit_shared_study_v1.py')]
        def exe(self): return current_executable
    real_process = supervision.psutil.Process()
    monkeypatch.setattr(supervision.psutil, 'Process', lambda pid=None: Parent() if pid == 4242 else real_process)
    monkeypatch.setattr(supervision.os, 'getppid', lambda: 4242)
    monkeypatch.setattr(supervision.psutil, 'virtual_memory', lambda: type('M', (), {'available': 100 * 1024**3})())
    monkeypatch.setattr(supervision.shutil, 'disk_usage', lambda _: type('D', (), {'free': 100 * 1024**3})())
    registry = tmp_path / 'registry'; registry.mkdir()
    (registry / 'registry.json').write_text('{}'); (registry / 'completion.json').write_text('{}')
    root = tmp_path / 'study'; root.mkdir()
    tasks = supervisor.phases(root, registry, 0)
    manifest = {'format': 'bounded_shared_crossfit_training_study_v1', 'registry_path': str(registry),
                'registry_sha256': digest(registry / 'registry.json'),
                'registry_completion_sha256': digest(registry / 'completion.json'), 'source_sha256': {},
                'cache_mib': 0, 'min_disk_bytes': 26 * 1024**3, 'min_available_ram_bytes': 4 * 1024**3,
                'max_process_rss_bytes': 6 * 1024**3, 'supervisor_pid': 4242,
                'supervisor_create_time': 123.5, 'phases': [name for name, _, _ in tasks]}
    snapshot = root / 'source_snapshot'; snapshot.mkdir()
    (snapshot / 'snapshot_manifest.json').write_text(json.dumps({'source_sha256': {}}))
    manifest['source_snapshot_sha256'] = digest(snapshot / 'snapshot_manifest.json')
    (root / 'run_manifest.json').write_text(json.dumps(manifest))
    lock_path = tmp_path / '.source_group_capture_resume_v1.lock'
    with lock_path.open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert supervision.assert_supervised(root, 'verification', root / 'verification') == manifest
        with pytest.raises(ValueError, match='Child phase'):
            supervision.assert_supervised(root, 'probe', root / 'probe')
        artifact = root / 'verification' / 'verification.json'; artifact.parent.mkdir()
        artifact.write_text('{}'); log = root / 'verification.log'; log.write_text('completed')
        row = {'phase': 'verification', 'passed': True, 'artifact_path': str(artifact),
               'artifact_sha256': digest(artifact), 'log_sha256': digest(log)}
        (root / 'phases.jsonl').write_text(json.dumps(row) + '\n')
        assert supervision.assert_supervised(root, 'probe', root / 'probe') == manifest
        log.write_text('tampered')
        with pytest.raises(ValueError, match='Frozen phase artifact changed'):
            supervision.assert_supervised(root, 'probe', root / 'probe')
    with pytest.raises(ValueError, match='lock is not held'):
        supervision.assert_supervised(root, 'verification', root / 'verification')


def test_source_snapshot_detects_changed_archived_bytes(tmp_path, monkeypatch):
    project = tmp_path / 'project'; project.mkdir(); source = project / 'module.py'; source.write_text('original')
    monkeypatch.setattr(supervisor, 'ROOT', project)
    destination = tmp_path / 'run' / 'source_snapshot'
    source_map = {'module.py': digest(source)}
    snapshot_sha = supervisor.snapshot_sources(destination, source_map)
    manifest = {'source_snapshot_sha256': snapshot_sha, 'source_sha256': source_map}
    supervision.check_source_snapshot(destination.parent, manifest)
    (destination / 'module.py').write_text('changed')
    with pytest.raises(ValueError, match='snapshot content'):
        supervision.check_source_snapshot(destination.parent, manifest)


@pytest.mark.parametrize('peak, cap', [(None, 200), (201, 200), (100, None), (0, 200)])
def test_independent_probe_auditor_rejects_missing_or_excess_rss(tmp_path, monkeypatch, peak, cap):
    monkeypatch.setattr(probe_auditor, 'BASE', tmp_path); monkeypatch.setattr(probe_auditor, 'ROOT', tmp_path)
    monkeypatch.setattr(probe_auditor.worker, 'sources', lambda: {})
    scripts = tmp_path / 'scripts'; scripts.mkdir()
    runner = scripts / 'probe_bounded_crossfit_shared_v1.py'; runner.write_text('probe')
    folder = tmp_path / 'registry'; folder.mkdir()
    registry = {'preflight_sha256': 'pre', 'environment': {}}
    (folder / 'registry.json').write_text(json.dumps(registry))
    evidence = tmp_path / 'verification.json'
    rebuilt = {'passed': True, 'registry_sha256': digest(folder / 'registry.json')}
    evidence.write_text(json.dumps(rebuilt | {'label_guard': {'open_attempts': 0}}))
    monkeypatch.setattr(probe_auditor.verifier, 'verify', lambda _: rebuilt)
    probe = tmp_path / 'probe.json'
    record = {'passed': True, 'format': 'bounded_shared_six_arm_probe_v1', 'device': 'mps',
              'registry_sha256': digest(folder / 'registry.json'), 'verification_sha256': digest(evidence),
              'preflight_sha256': 'pre', 'runner_sha256': digest(runner), 'source_sha256': {},
              'environment': {}, 'label_guard': {'open_attempts': 0},
              'peak_process_rss_bytes': peak, 'max_process_rss_bytes': cap}
    probe.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='resource evidence'):
        probe_auditor.audit(folder, evidence, probe)


def test_independent_probe_auditor_checks_saved_weight_update(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_auditor, 'BASE', tmp_path); monkeypatch.setattr(probe_auditor, 'ROOT', tmp_path)
    monkeypatch.setattr(probe_auditor.worker, 'sources', lambda: {})
    scripts = tmp_path / 'scripts'; scripts.mkdir()
    runner = scripts / 'probe_bounded_crossfit_shared_v1.py'; runner.write_text('probe')
    folder = tmp_path / 'registry'; folder.mkdir()
    registry = {'preflight_sha256': 'c' * 64, 'environment': {'device': 'mps'},
                'handoff_sha256': 'h' * 64, 'backbone_sha256': 'a' * 64, 'feature_pipeline_sha256': 'b' * 64,
                'initial_state_sha256': {arm: 'i' * 64 for arm in worker.ARMS},
                'samples': [{'sequence': 'seq6', 'frame': 2}, {'sequence': 'seq23', 'frame': 3}]}
    (folder / 'registry.json').write_text(json.dumps(registry)); registry_sha = digest(folder / 'registry.json')
    evidence = tmp_path / 'verification.json'; rebuilt = {'passed': True, 'registry_sha256': registry_sha}
    evidence.write_text(json.dumps(rebuilt | {'label_guard': {'open_attempts': 0}}))
    monkeypatch.setattr(probe_auditor.verifier, 'verify', lambda _: rebuilt)
    probe_folder = tmp_path / 'probe'; probe_folder.mkdir(); checks = []
    authorization = {'preflight_sha256': registry['preflight_sha256'], 'handoff_sha256': registry['handoff_sha256'],
                     'registry_sha256': registry_sha, 'source_sha256': {}}
    for sequence, frame in (('seq6', 2), ('seq23', 3)):
        for arm in worker.ARMS:
            model = worker.model_for(arm)
            with torch.no_grad(): model.head[-1].bias.fill_(0.01)
            path = probe_folder / f'{sequence}_{arm}.pt'
            torch.save(worker.save_payload(model, arm=arm, fold=1, registry=registry,
                                           authorization=authorization, steps=1), path)
            checks.append({'sequence': sequence, 'frame': frame, 'arm': arm, 'optimizer_updated': True,
                'gradient_l2': 1.0, 'head_bias_gradient_abs': .1, 'initial_state_sha256': 'i' * 64,
                'video_shape': [720, 834] if sequence == 'seq23' else [3, 4], 'synthetic_loss': 1.0,
                'checkpoint_path': str(path), 'checkpoint_sha256': digest(path)})
    probe = probe_folder / 'completion.json'
    record = {'passed': True, 'format': 'bounded_shared_six_arm_probe_v1', 'device': 'mps',
              'registry_sha256': registry_sha, 'verification_sha256': digest(evidence),
              'preflight_sha256': registry['preflight_sha256'], 'runner_sha256': digest(runner), 'source_sha256': {},
              'environment': registry['environment'], 'label_guard': {'open_attempts': 0},
              'peak_process_rss_bytes': 100, 'max_process_rss_bytes': 200, 'checks': checks}
    probe.write_text(json.dumps(record))
    assert probe_auditor.audit(folder, evidence, probe)['audited_checks'] == 12
    target = Path(checks[0]['checkpoint_path']); payload = torch.load(target, weights_only=True)
    payload['state_dict']['head.2.bias'].zero_(); torch.save(payload, target)
    checks[0]['checkpoint_sha256'] = digest(target); probe.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='no head update'):
        probe_auditor.audit(folder, evidence, probe)
