"""CPU-only regression tests for archived-A observer-only capture."""
import json
from pathlib import Path

import pytest

from scripts import run_crossfit_observer_capture_study_v2 as study
from scripts import crossfit_observer_capture_shared_v2 as shared
from scripts.memory_token_io import digest


def test_one_observed_pass_no_fresh_full_reference(tmp_path):
    with pytest.raises(ValueError,match='Unknown capture stage'):
        study.phases('short',tmp_path)
    tasks=study.phases('full',tmp_path,tmp_path/'gate.json','seq6',143)
    assert [row[0] for row in tasks]==['seq6_capture','seq6_audit']
    assert '--reference' not in tasks[0][1] and '--reference' not in tasks[1][1]
    assert tasks[0][1][tasks[0][1].index('--remaining-reads')+1]=='143'
    with pytest.raises(ValueError,match='one planned sequence'):
        study.phases('full',tmp_path,tmp_path/'gate.json')


def test_v2_binds_frozen_short_gate_sources_and_exclusive_lock():
    hashes=shared.source_hashes()
    assert all(hashes[k]==v for k,v in shared.v1_source_hashes().items())
    for key in ('scripts/resume_source_group_capture_v1.py',
                'scripts/run_crossfit_observer_capture_v2.py',
                'scripts/audit_crossfit_observer_full_sequence_v2.py',
                'scripts/run_crossfit_observer_capture_study_v2.py'):
        assert hashes[key]==digest(shared.ROOT/key)


def test_resume_verifies_archived_reference_and_capture_tensor_integrity(monkeypatch,tmp_path):
    monkeypatch.setattr(study,'OUT',tmp_path)
    gate=tmp_path/'gate.json';gate.write_text('{}')
    baseline=tmp_path/'archived_A.json';baseline.write_text('{}')
    plan={'fresh_reads':[{'sequence':'seq6','frame':24},{'sequence':'seq6','frame':90},
                         {'sequence':'seq8','frame':8}],
          'prompts':{'seq6':{'baseline_completion_path':str(baseline),'baseline_completion_sha256':digest(baseline)}}}
    sources={'source':'digest'};batch=tmp_path/'batch'
    for name in ('capture','audit'):(batch/'seq6'/name).mkdir(parents=True)
    cap=batch/'seq6/capture/receipt.json';observations=[]
    for frame in (24,90):
        row={'frame':frame}
        for key in ('feature','snapshot'):
            file=cap.parent/f'{key}_{frame}.pt';file.write_bytes(f'{key}{frame}'.encode())
            row[key+'_file']=str(file);row[key+'_sha256']=digest(file)
        observations.append(row)
    cap.write_text(json.dumps({'passed':True,'sequence':'seq6','mode':'capture','observations':observations}))
    audit=batch/'seq6/audit/audit.json';audit.write_text(json.dumps({
        'passed':True,'sequence':'seq6','plan_sha256':study.PLAN_SHA,'source_sha256':sources,
        'engineering_audit_sha256':digest(gate),'auditor_sha256':digest(study.FULL_AUDITOR),
        'label_guard':{'open_attempts':0},'reference_kind':'source_bound_archived_A',
        'reference_receipt_path':str(baseline),'reference_receipt_sha256':digest(baseline),
        'capture_receipt_sha256':digest(cap),'target_frames':[24,90]}))
    rows=[]
    for phase,file in [('seq6_capture',cap),('seq6_audit',audit)]:
        log=batch/(phase+'.log');log.write_text('done')
        rows.append({'phase':phase,'passed':True,'artifact_path':str(file),
                     'artifact_sha256':digest(file),'log_sha256':digest(log)})
    (batch/'phases.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    accepted=study.checked_audits([audit],plan,sources,gate)
    assert study.remaining_count(plan,accepted)==1
    tensor=Path(observations[0]['feature_file']);tensor.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='tensor missing or changed'):
        study.checked_audits([audit],plan,sources,gate)
    tensor.unlink()
    with pytest.raises(FileNotFoundError):study.checked_audits([audit],plan,sources,gate)
    baseline.write_text('{"tampered":true}')
    with pytest.raises(ValueError,match='Prior audit'):
        study.checked_audits([audit],plan,sources,gate)


def test_audit_accounts_actual_written_only_and_preserves_other_reservations(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import shutil
    remaining=143;budget=12*shared.PER_READ;written=7*shared.PER_READ
    required=shared.GUARD+shared.MARGIN+remaining*shared.PER_READ-written
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=required))
    space=study.headroom_with_written(tmp_path,remaining,written,budget)
    assert space['minimum_bytes']==required and space['written_payload_bytes_accounted']==written
    with pytest.raises(ValueError,match='Insufficient headroom'):
        study.headroom_with_written(tmp_path,remaining,0,budget)
    # Even excess bytes cannot release the reservation for other sequences.
    with pytest.raises(ValueError,match='Insufficient headroom'):
        study.headroom_with_written(tmp_path,remaining,20*shared.PER_READ,budget-6*shared.PER_READ)


def test_audit_subprocess_uses_written_payload_reservation(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import shutil
    remaining=143;written=7*shared.PER_READ;budget=12*shared.PER_READ
    required=shared.GUARD+shared.MARGIN+remaining*shared.PER_READ-written
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=required))
    monkeypatch.setattr(study.subprocess,'Popen',lambda *a,**k:SimpleNamespace(wait=lambda **kw:0))
    study.run_phase(['noop'],tmp_path/'audit.log',None,remaining,written=written,sequence_budget=budget)
    with pytest.raises(ValueError,match='headroom lost after child'):
        study.run_phase(['noop'],tmp_path/'unreserved.log',None,remaining,written=0,sequence_budget=budget)
