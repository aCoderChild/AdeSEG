import json
from pathlib import Path

import pytest

from scripts import reconcile_additional_author_audit_v3 as target
from scripts.memory_token_io import digest


def test_reconciliation_binds_failed_log_original_score_and_corrected_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(target,'EXPERIMENT_ROOT',tmp_path)
    batch=tmp_path/'batch';batch.mkdir();(batch/'scores').mkdir();(batch/'verification').mkdir()
    run_map=batch/'run_map.json';run_map.write_text('{}')
    manifest=batch/'run_manifest.json';manifest.write_text(json.dumps({'sources':{'auditor':digest(target.ROOT/'scripts/audit_additional_author_comparators_v2.py')}}))
    failed=batch/'audit.log';failed.write_text("metadata[seq]\nTypeError: list indices must be integers or slices, not str\n")
    proof={'passed':True,'label_guard':{'open_attempts':0},'selection':['seq2'],
           'run_map_sha256':digest(run_map),'launch_manifest_sha256':'manifest','receipts':{},
           'references':{'seq2':{}},'inputs':{'seq2':{'images':['a','b']}}}
    verification=batch/'verification/verification.json';verification.write_text(json.dumps(proof))
    score=batch/'scores/results.json';score.write_text(json.dumps({'passed':True,'verification_sha256':digest(verification),
                                                                     'run_map_sha256':digest(run_map)}))
    phases=[{'phase':'verify','passed':True,'artifact_sha256':digest(verification)},
            {'phase':'score','passed':True,'artifact_sha256':digest(score)}]
    (batch/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in phases))
    audit=tmp_path/'corrected.json';audit.write_text(json.dumps({'passed':True,'error_count':0,'errors':[],
        'auditor_sha256':digest(target.ROOT/'scripts/audit_additional_author_comparators_v3.py'),
        'run_map_sha256':digest(run_map),'verification_sha256':digest(verification),'score_sha256':digest(score),
        'prelabel_guard':{'open_attempts':0},'label_files_checked':2,'prediction_files_checked':6}))
    monkeypatch.setattr(target,'validate_frozen',lambda *_args,**_kwargs:proof)
    result=target.reconcile(batch,audit)
    assert result['passed'] and result['expected_labels']==2 and result['expected_predictions']==6
    phases[0]['artifact_sha256']='0'*64
    (batch/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in phases))
    with pytest.raises(ValueError,match='phase-frozen'):
        target.reconcile(batch,audit)
    phases[0]['artifact_sha256']=digest(verification);phases[1]['artifact_sha256']='0'*64
    (batch/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in phases))
    with pytest.raises(ValueError,match='phase-frozen'):
        target.reconcile(batch,audit)
    phases[1]['artifact_sha256']=digest(score)
    (batch/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in phases))
    failed.write_text('different failure')
    with pytest.raises(ValueError,match='expected V2|Expected V2'):
        target.reconcile(batch,audit)
