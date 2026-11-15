import json
from pathlib import Path

import pytest

from scripts import audit_boundary_video_short_v1 as audit


def test_rejects_failed_gate_without_opening_run_artifacts(tmp_path):
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps({'passed': False}))
    with pytest.raises(ValueError, match='Short gate failed'):
        audit.audit(gate)


def test_rejects_changed_gate_source_before_opening_run_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    runner = tmp_path / 'scripts/run_boundary_refiner_priority_v1.py'
    runner.parent.mkdir()
    runner.write_text('actual')
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps({'passed': True, 'runner_sha256': 'bad'}))
    with pytest.raises(ValueError, match='Gate source differs'):
        audit.audit(gate)


def test_digest_and_explicit_require(tmp_path):
    path = tmp_path / 'x'
    path.write_bytes(b'example')
    assert audit.sha(path) == '50d858e0985ecc7f60418aaf0cc5ab587f42c2570a884095a9e8ccacd0f6545c'
    with pytest.raises(ValueError, match='missing'):
        audit.require(False, 'missing')
