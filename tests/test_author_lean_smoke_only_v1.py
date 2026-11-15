import pytest

from scripts import run_author_lean_smoke_only_v1 as short


def test_short_launcher_stops_after_exactly_two_smokes_and_gate(tmp_path):
    phases = short.short_phases(tmp_path)
    assert [name for name, _ in phases] == ['smoke_seq6', 'smoke_seq23', 'gate']
    assert all('full_' not in name and name not in ('verify', 'score', 'audit') for name, _ in phases)


def test_short_launcher_rejects_changed_original_phase_order(tmp_path, monkeypatch):
    monkeypatch.setattr(short, 'commands', lambda _: [('smoke_seq6', []), ('full_seq6', []), ('gate', [])])
    with pytest.raises(ValueError, match='phase order'):
        short.short_phases(tmp_path)


@pytest.mark.parametrize('failure_index', [0, 1, 2])
def test_each_failure_stops_later_phases_and_never_runs_full(tmp_path, monkeypatch, failure_index):
    names = ['smoke_seq6', 'smoke_seq23', 'gate']
    monkeypatch.setattr(short, 'short_phases', lambda _: [(name, [name]) for name in names])
    monkeypatch.setattr(short, 'short_available', lambda _: 30 * 1024**3)
    monkeypatch.setattr(short, 'artifact', lambda folder, name: folder/(name+'.artifact'))
    calls = []
    def fake_run(cmd, log, folder, errors):
        calls.append(cmd[0])
        log.write_text('partial or completed log')
        if len(calls)-1 == failure_index:
            raise RuntimeError('failure')
        (folder/(cmd[0]+'.artifact')).write_text('output')
    monkeypatch.setattr(short, 'run_short_phase', fake_run)
    with pytest.raises(RuntimeError, match='failure'):
        short.execute_phases(tmp_path, lambda: None, [])
    assert calls == names[:failure_index+1]
    assert not any('full' in name for name in calls)
    assert (tmp_path/(names[failure_index]+'.log')).exists()
    ledger = tmp_path/'phases.jsonl'
    assert len(ledger.read_text().splitlines()) == failure_index if ledger.exists() else failure_index == 0


def test_space_loss_terminates_child_and_keeps_partial_log(tmp_path, monkeypatch):
    import subprocess
    calls = []
    class Process:
        def wait(self, timeout=None):
            if not calls:
                calls.append('wait')
                raise subprocess.TimeoutExpired('mock', timeout)
            return 0
        def terminate(self): calls.append('terminate')
    monkeypatch.setattr(short.subprocess, 'Popen', lambda *a, **k: Process())
    checks = iter([30 * 1024**3, RuntimeError('reserve lost')])
    def available(_):
        value = next(checks)
        if isinstance(value, Exception): raise value
        return value
    monkeypatch.setattr(short, 'short_available', available)
    log = tmp_path/'partial.log'
    with pytest.raises(RuntimeError, match='reserve lost'):
        short.run_short_phase(['fake'], log, tmp_path, [])
    assert 'terminate' in calls and log.exists()


def test_handoff_binds_short_launcher_and_phase_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(short, 'artifact', lambda folder, name: folder/(name+'.json'))
    (tmp_path/'phases.jsonl').write_text('phase evidence')
    (tmp_path/'gate.json').write_text('gate evidence')
    own = short.digest(short.Path(short.__file__))
    short.write_handoff(tmp_path, own, 'manifest digest')
    import json
    handoff = json.loads((tmp_path/'short_handoff.json').read_text())
    assert handoff['launcher_sha256'] == short.digest(tmp_path/'short_only_launcher.py') == own
    assert handoff['gate_sha256'] == short.digest(tmp_path/'gate.json')
    assert handoff['full_resume_requires_independent_handoff_validation']
