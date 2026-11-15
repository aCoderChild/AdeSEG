import json
from pathlib import Path
from types import SimpleNamespace

from scripts import observe_author_matching_run as observer


def test_recognizes_protocol_runner_and_rejects_wrong_pid_identity(tmp_path, monkeypatch):
    runner = str(observer.ROOT / 'scripts/run_pitome_causal_diagnostic.py')
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'protocol': {'runner': 'scripts/run_pitome_causal_diagnostic.py'},
                                'counts': {'targets': 318}}))
    (tmp_path / 'queue_setup.json').write_text(json.dumps({'plan': str(plan)}))
    (tmp_path / 'progress.json').write_text(json.dumps({'phase': 'cache_gate', 'pid': 42}))
    monkeypatch.setattr(observer, 'validate_output', Path)
    command = f'42 00:10 100.0 2048 Python -u {runner} --plan {plan}'
    monkeypatch.setattr(observer.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=command))
    assert observer.observe(tmp_path)['child']['live']
    assert observer.observe(tmp_path)['state'] == 'running'
    command = '42 00:10 100.0 2048 Python unrelated.py'
    assert not observer.observe(tmp_path)['child']['live']
    assert observer.observe(tmp_path)['child']['pid_identity_mismatch']
