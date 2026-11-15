import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_source_group_refiner_study_v1 as runner


def test_phases_have_required_order_and_unique_outputs(tmp_path):
    phases = runner.make_phases(tmp_path)
    assert [name for name, _ in phases] == [
        'manifest', 'plan', 'parity', 'train_region', 'train_global',
        'train_no_contrast', 'train_mask_only', 'verify', 'score',
    ]
    paths = [command[command.index('--output') + 1] for _, command in phases]
    assert len(paths) == len(set(paths))
    assert all(Path(path).is_relative_to(tmp_path) for path in paths)
    assert '--verification' in phases[-1][1]
    assert phases[-1][1][phases[-1][1].index('--verification') + 1] == str(tmp_path/'verification')


def test_map_requires_all_audited_sequences():
    with pytest.raises(ValueError, match='nine audited'):
        runner.make_capture_map({})


def test_map_uses_auditor_paths_and_rejects_changed_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'SEQUENCES', ('seq2',))
    monkeypatch.setattr(runner, 'resolve_capture_map', lambda mapping: mapping)
    reference = tmp_path/'reference'/'receipt.json'
    capture = tmp_path/'capture'/'receipt.json'
    audit = tmp_path/'audit'/'verification.json'
    audit.parent.mkdir()
    audit.write_text(json.dumps({'passed': True, 'sequence': 'seq2',
                                 'reference_receipt_path': str(reference),
                                 'capture_receipt_path': str(capture)}))
    completed = {'seq2': [{'audit_path': str(audit), 'audit_sha256': runner.digest(audit)}]}
    mapping = runner.make_capture_map(completed)
    assert mapping['seq2'] == {'reference': str(reference.parent), 'capture': str(capture.parent),
                               'audit': str(audit.parent)}
    audit.write_text('{}')
    with pytest.raises(ValueError, match='audit changed'):
        runner.make_capture_map(completed)


def test_free_space_guard_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=runner.GUARD_BYTES-1))
    with pytest.raises(RuntimeError, match='below'):
        runner.checked_space(tmp_path)


def test_failed_first_phase_never_runs_training_or_scoring(tmp_path, monkeypatch):
    pre, freeze = tmp_path/'preflight', tmp_path/'freeze'
    pre.mkdir(); freeze.mkdir()
    (pre/'manifest.json').write_text('{}'); (freeze/'source_freeze.json').write_text('{}')
    gate, checkpoint = tmp_path/'gate.json', tmp_path/'checkpoint.pt'
    gate.write_text('{}'); checkpoint.write_bytes(b'weights')
    monkeypatch.setattr(runner, 'PRE', pre)
    monkeypatch.setattr(runner, 'FREEZE', freeze)
    monkeypatch.setattr(runner, 'GATE', gate)
    monkeypatch.setattr(runner, 'CHECKPOINT', checkpoint)
    monkeypatch.setattr(runner, 'dependencies', lambda: {'model': 'unchanged'})
    monkeypatch.setattr(runner, 'checked_space', lambda _: runner.GUARD_BYTES+1)
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(runner.subprocess, 'run', fail)
    run = tmp_path/'run'; run.mkdir()
    with pytest.raises(RuntimeError, match='manifest failed'):
        runner.execute(run, {}, {})
    assert len(calls) == 1
    assert calls[0][1].endswith('prepare_source_group_refiner_manifest_v1.py')
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    assert [x['status'] for x in events] == ['started', 'failed']
    assert not (run/'scores').exists()


def test_source_change_aborts_before_subprocess(tmp_path, monkeypatch):
    pre, freeze = tmp_path/'preflight', tmp_path/'freeze'
    pre.mkdir(); freeze.mkdir()
    (pre/'manifest.json').write_text('{}'); (freeze/'source_freeze.json').write_text('{}')
    gate, checkpoint = tmp_path/'gate.json', tmp_path/'checkpoint.pt'
    gate.write_text('{}'); checkpoint.write_bytes(b'weights')
    monkeypatch.setattr(runner, 'PRE', pre)
    monkeypatch.setattr(runner, 'FREEZE', freeze)
    monkeypatch.setattr(runner, 'GATE', gate)
    monkeypatch.setattr(runner, 'CHECKPOINT', checkpoint)
    results = iter([{'model': 'old'}, {'model': 'new'}])
    monkeypatch.setattr(runner, 'dependencies', lambda: next(results))
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch'))
    run = tmp_path/'run'; run.mkdir()
    with pytest.raises(RuntimeError, match='dependencies changed'):
        runner.execute(run, {}, {})
    assert not (run/'manifest').exists()


def prepare_fake_run(tmp_path, monkeypatch):
    pre, freeze = tmp_path/'preflight', tmp_path/'freeze'
    pre.mkdir(); freeze.mkdir()
    (pre/'manifest.json').write_text('{}'); (freeze/'source_freeze.json').write_text('{}')
    gate, checkpoint = tmp_path/'gate.json', tmp_path/'checkpoint.pt'
    gate.write_text('{}'); checkpoint.write_bytes(b'weights')
    for name, value in (('PRE', pre), ('FREEZE', freeze), ('GATE', gate), ('CHECKPOINT', checkpoint)):
        monkeypatch.setattr(runner, name, value)
    monkeypatch.setattr(runner, 'dependencies', lambda: {'model': 'unchanged'})
    monkeypatch.setattr(runner, 'checked_space', lambda _: runner.GUARD_BYTES+1)
    run = tmp_path/'run'; run.mkdir()
    return run


def test_failed_verification_never_starts_scorer_and_sets_mps_env(tmp_path, monkeypatch):
    run = prepare_fake_run(tmp_path, monkeypatch)
    calls = []
    def call(command, **kwargs):
        calls.append((command, kwargs['env']))
        return SimpleNamespace(returncode=7 if 'verify_source_group_refiner_v1.py' in command[1] else 0)
    monkeypatch.setattr(runner.subprocess, 'run', call)
    with pytest.raises(RuntimeError, match='verify failed'):
        runner.execute(run, {}, {})
    assert len(calls) == 8
    assert all(env['PYTORCH_ENABLE_MPS_FALLBACK'] == '0' and env['PYTORCH_MPS_FAST_MATH'] == '0'
               for _, env in calls)
    assert not any('score_source_group_refiner_v1.py' in command[1] for command, _ in calls)


def test_monitor_failure_after_subprocess_blocks_following_phase(tmp_path, monkeypatch):
    run = prepare_fake_run(tmp_path, monkeypatch)
    monitor_errors = []
    def monitor(_run, stop, ready, errors):
        monitor_errors.append(errors)
        ready.set()
        stop.wait()
    def call(*_args, **_kwargs):
        monitor_errors[0].append('disk monitor failed')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner, 'monitor_space', monitor)
    monkeypatch.setattr(runner.subprocess, 'run', call)
    with pytest.raises(RuntimeError, match='phase outputs may exist'):
        runner.execute(run, {}, {})
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    assert [event['status'] for event in events] == ['started', 'monitor_failed_after_phase']
    assert not (run/'plan.log').exists()


def test_monitor_failure_during_final_join_is_reported(tmp_path, monkeypatch):
    run = prepare_fake_run(tmp_path, monkeypatch)
    calls = []
    def monitor(_run, stop, ready, errors):
        ready.set()
        stop.wait()
        errors.append('monitor failed while stopping')
    def call(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner, 'monitor_space', monitor)
    monkeypatch.setattr(runner.subprocess, 'run', call)
    with pytest.raises(RuntimeError, match='after the final phase'):
        runner.execute(run, {}, {})
    assert len(calls) == 9
    assert 'score_source_group_refiner_v1.py' in calls[-1][1]
