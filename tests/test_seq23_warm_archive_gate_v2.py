"""CPU-only tests for uniform archived-mask warm initialization."""
import json
from pathlib import Path
import pytest
from scripts import run_seq23_warm_archive_gate_v2 as runner
from scripts import audit_seq23_warm_archive_gate_v2 as audit
from scripts import run_seq23_warm_archive_gate_study_v2 as study
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def test_seq23_gate_runs_only_mps_reference_observer_and_audit(tmp_path):
    tasks=study.phase_tasks(tmp_path)
    assert [x[0] for x in tasks]==['seq23_reference','seq23_capture','audit']
    assert [x[3] for x in tasks]==[0,1,0]
    assert all('--sequence' in x[1] for x in tasks[:2])


def test_partial_seq3_seq6_compatibility_is_bound_and_label_free():
    guard=install_mask_open_guard(runner.DATA);part=audit.check_partial()
    assert part['passed'] and part['sequences']['seq3']['exact_compatibility']
    assert part['sequences']['seq6']['exact_compatibility'] and guard['open_attempts']==0


def test_changed_partial_audit_cannot_authorize_combined_gate(monkeypatch,tmp_path):
    file=tmp_path/'audit.json';file.write_text('{}')
    (tmp_path/'completion.json').write_text(json.dumps({'passed':True,'audit_sha256':'different','label_guard':{'open_attempts':0}}))
    monkeypatch.setattr(audit,'PARTIAL',file)
    with pytest.raises(ValueError,match='partial evidence changed'):audit.check_partial()


def test_gate_binds_indirect_project_imports():
    from scripts.local_import_closure_v1 import local_import_closure
    assert 'scripts/audit_crossfit_observer_capture_v1.py' in runner.SOURCES
    assert 'scripts/local_import_closure_v1.py' in runner.SOURCES
    assert local_import_closure(runner.ROOT,runner.SOURCES).issubset(set(runner.SOURCES))
