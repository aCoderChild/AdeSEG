"""CPU-only tests for reviewed cross-fit observer/capture mechanics."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import crossfit_observer_capture_shared_v1 as shared
from scripts import run_crossfit_observer_capture_study_v1 as study
from scripts import run_crossfit_observer_capture_v1 as runner
from scripts import audit_crossfit_observer_capture_v1 as auditor
from scripts.memory_token_io import digest


def test_phase_schedule_requires_short_audit_and_orders_seq6_first(tmp_path):
    short=study.phases('short',tmp_path)
    assert len(short)==5 and short[-1][0]=='engineering_audit'
    assert [x[0] for x in short[:2]]==['seq6_reference','seq6_capture']
    with pytest.raises(ValueError,match='short gate'):study.phases('full',tmp_path)
    full=study.phases('full',tmp_path,tmp_path/'audit.json','seq6',143)
    assert [x[0] for x in full]==['seq6_reference','seq6_capture','seq6_audit']
    assert '--remaining-reads' in full[0][1] and '143' in full[0][1]
    with pytest.raises(ValueError,match='one planned sequence'):
        study.phases('full',tmp_path,tmp_path/'audit.json')


def test_headroom_reserves_all_remaining_payload_and_margin(monkeypatch,tmp_path):
    minimum=shared.GUARD+shared.MARGIN+143*shared.PER_READ
    monkeypatch.setattr(shared.shutil,'disk_usage',lambda _:SimpleNamespace(free=minimum))
    assert shared.require_headroom(tmp_path,143)['minimum_bytes']==minimum
    monkeypatch.setattr(shared.shutil,'disk_usage',lambda _:SimpleNamespace(free=minimum-1))
    with pytest.raises(ValueError,match='Insufficient headroom'):shared.require_headroom(tmp_path,143)


def test_short_gate_outputs_do_not_reduce_future_full_payload_reserve(monkeypatch,tmp_path):
    monkeypatch.setattr(study.subprocess,'Popen',lambda *a,**k:SimpleNamespace(wait=lambda **kw:0))
    calls=iter([0,0,shared.PER_READ])
    monkeypatch.setattr(study,'tensor_bytes',lambda _:next(calls))
    remaining=143;free=shared.GUARD+shared.MARGIN+(remaining-1)*shared.PER_READ
    monkeypatch.setattr(shared.shutil,'disk_usage',lambda _:SimpleNamespace(free=free))
    with pytest.raises(ValueError,match='headroom lost after child'):
        study.run_phase(['noop'],tmp_path/'short.log',tmp_path,remaining,deduct_output=False)
    study.run_phase(['noop'],tmp_path/'full.log',tmp_path,remaining,deduct_output=True)


def test_full_capture_requires_mps_flags_and_short_audit_before_model(monkeypatch,tmp_path):
    for key in ('PYTHONHASHSEED','PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH'):
        monkeypatch.setenv(key,'0')
    monkeypatch.setattr(runner.sys,'argv',['capture','--sequence','seq6','--mode','reference','--limit','0',
                                           '--remaining-reads','143','--output',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='independently audited short gate'):runner.main()
    assert not (tmp_path/'out').exists()


def test_baseline_state_and_trace_tampering_is_rejected(tmp_path):
    trace={'memory_events':[{'a':1}],'positions':[{'p':1}],'selections':[]};rec={}
    for key,rows in trace.items():
        path=tmp_path/(key+'.json');path.write_text(json.dumps(rows));rec[key+'_sha256']=digest(path)
    rec['post_prompt_state']=[{'frame':1,'mask_sha256':'a'},{'frame':2,'mask_sha256':'b'}]
    baseline=tmp_path/'completion.json';baseline.write_text('{}')
    runner.check_baseline_rows(rec['post_prompt_state'],trace,baseline,rec,short=False)
    with pytest.raises(ValueError,match='native state'):
        runner.check_baseline_rows([{'frame':1,'mask_sha256':'bad'}],trace,baseline,rec,short=True)
    with pytest.raises(ValueError,match='positions'):
        runner.check_baseline_rows(rec['post_prompt_state'],{**trace,'positions':[{'p':2}]},baseline,rec,short=False)


def test_observer_tensor_comparison_is_exact_and_checks_nonfinite():
    a={'x':[torch.tensor([1.,2.]),None]};b={'x':[torch.tensor([1.,2.]),None]}
    assert auditor.tensor_tree_equal(a,b) and auditor.tensor_tree_finite(a)
    b['x'][0][1]=3
    assert not auditor.tensor_tree_equal(a,b)
    assert not auditor.tensor_tree_finite({'x':torch.tensor([float('nan')])})


def test_short_gate_rejects_reused_process_and_changed_observer(monkeypatch,tmp_path):
    monkeypatch.setattr(auditor,'BASE',tmp_path)
    monkeypatch.setattr(auditor,'sequence_inputs',lambda plan,seq:([],[],0,[],None,tmp_path/'baseline',{},{}))
    monkeypatch.setattr(auditor,'historical_seq6_comparison',lambda *_:{'frame':1})
    items={}
    for index,seq in enumerate(shared.SHORT):
        for offset,name in enumerate(('reference','capture')):
            folder=tmp_path/seq/name;folder.mkdir(parents=True)
            items[str(folder)]={'folder':str(folder),'receipt_sha256':f'{seq}-{name}',
                'process_id':index*2+offset+1,'run_uuid':f'{seq}-{name}',
                'reference_receipt_sha256':None if name=='reference' else f'{seq}-reference',
                'trace':{'events':[1]},'rows':[{'frame':0}],
                'tensors':[] if name=='reference' else [[torch.tensor([1.])]]}
    monkeypatch.setattr(auditor,'checked_run',lambda folder,**kwargs:items[str(Path(folder))])
    monkeypatch.setattr(auditor,'phase_artifact',lambda root,name,path:items[str(path.parent)]['receipt_sha256'])
    assert auditor.audit_engineering(tmp_path,{},{} )['passed']
    items[str(tmp_path/'seq23'/'capture')]['process_id']=items[str(tmp_path/'seq23'/'reference')]['process_id']
    with pytest.raises(ValueError,match='fresh processes'):auditor.audit_engineering(tmp_path,{}, {})
    items[str(tmp_path/'seq23'/'capture')]['process_id']=99
    items[str(tmp_path/'seq23'/'capture')]['trace']={'events':[2]}
    with pytest.raises(ValueError,match='trajectory or trace'):auditor.audit_engineering(tmp_path,{}, {})


def test_phase_binding_detects_changed_receipt_or_log(tmp_path):
    folder=tmp_path/'seq6'/'reference';folder.mkdir(parents=True)
    receipt=folder/'receipt.json';receipt.write_text('{}')
    log=tmp_path/'seq6_reference.log';log.write_text('finished')
    phase={'phase':'seq6_reference','passed':True,'artifact_path':str(receipt),
           'artifact_sha256':digest(receipt),'log_sha256':digest(log)}
    (tmp_path/'phases.jsonl').write_text(json.dumps(phase)+'\n')
    assert shared.phase_artifact(tmp_path,'seq6_reference',receipt)==digest(receipt)
    receipt.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='phase artifact changed'):
        shared.phase_artifact(tmp_path,'seq6_reference',receipt)


def test_engineering_audit_is_bound_to_its_own_completed_phase(monkeypatch,tmp_path):
    monkeypatch.setattr(auditor,'BASE',tmp_path)
    folder=tmp_path/'engineering_audit';folder.mkdir()
    evidence={'passed':True,'short_root':str(tmp_path),'label_guard':{'open_attempts':0}}
    file=folder/'audit.json';file.write_text(json.dumps(evidence))
    log=tmp_path/'engineering_audit.log';log.write_text('completed')
    phase={'phase':'engineering_audit','passed':True,'artifact_path':str(file),
           'artifact_sha256':digest(file),'log_sha256':digest(log)}
    (tmp_path/'phases.jsonl').write_text(json.dumps(phase)+'\n')
    monkeypatch.setattr(auditor,'audit_engineering',lambda *args:{'passed':True,'short_root':str(tmp_path)})
    assert auditor.verify_engineering_audit(file,{},{} )['passed']
    file.write_text(json.dumps({**evidence,'changed':True}))
    with pytest.raises(ValueError,match='phase artifact changed'):
        auditor.verify_engineering_audit(file,{}, {})


def test_real_v2_plan_and_seq23_prompt_roi_without_opening_labels():
    if not shared.PLAN_DIR.exists():pytest.skip('Reviewed Drive preflight is not mounted')
    code='''from scripts.crossfit_observer_capture_shared_v1 import validate_plan,sequence_inputs,archived_frame_metadata,DATA
from scripts.source_group_label_guard_v1 import install_mask_open_guard
guard=install_mask_open_guard(DATA);plan=validate_plan()
frames,_,start,targets,mask,_,_,prompt=sequence_inputs(plan,"seq23")
assert len(frames)==56 and start==0 and len(targets)==12 and mask.shape==(720,834)
assert prompt["prompt_provenance"]["roi_xyxy"]==[446,0,1280,720]
assert archived_frame_metadata(plan,"seq23")["sha256"]==prompt["source_index_metadata"]["sha256"]
assert guard["open_attempts"]==0
'''
    subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],check=True)


def test_resume_binds_audits_to_phases_and_counts_only_reviewed_reads(monkeypatch,tmp_path):
    monkeypatch.setattr(study,'OUT',tmp_path)
    engineering=tmp_path/'engineering.json';engineering.write_text('{}')
    plan={'fresh_reads':[{'sequence':'seq6','frame':24},{'sequence':'seq6','frame':90},
                         {'sequence':'seq8','frame':8}]}
    sources={'source':'digest'};batch=tmp_path/'batch'
    for name in ('reference','capture','audit'):
        (batch/'seq6'/name).mkdir(parents=True)
    ref=batch/'seq6/reference/receipt.json';cap=batch/'seq6/capture/receipt.json'
    ref.write_text('{}')
    observations=[]
    for frame in (24,90):
        item={'frame':frame}
        for key in ('feature','snapshot'):
            tensor=cap.parent/f'{key}_{frame}.pt';tensor.write_bytes(f'{key}{frame}'.encode())
            item[key+'_file']=str(tensor);item[key+'_sha256']=digest(tensor)
        observations.append(item)
    cap.write_text(json.dumps({'passed':True,'sequence':'seq6','mode':'capture','observations':observations}))
    audit=batch/'seq6/audit/audit.json'
    audit.write_text(json.dumps({'passed':True,'sequence':'seq6','plan_sha256':study.PLAN_SHA,
       'source_sha256':sources,'engineering_audit_sha256':digest(engineering),
       'auditor_sha256':digest(study.FULL_AUDITOR),'label_guard':{'open_attempts':0},
       'reference_receipt_sha256':digest(ref),'capture_receipt_sha256':digest(cap),'target_frames':[24,90]}))
    rows=[]
    for phase,artifact in [('seq6_reference',ref),('seq6_capture',cap),('seq6_audit',audit)]:
        log=batch/(phase+'.log');log.write_text('done')
        rows.append({'phase':phase,'passed':True,'artifact_path':str(artifact),
                     'artifact_sha256':digest(artifact),'log_sha256':digest(log)})
    (batch/'phases.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    accepted=study.checked_audits([audit],plan,sources,engineering)
    assert accepted=={'seq6':batch} and study.remaining_count(plan,accepted)==1
    with pytest.raises(ValueError,match='Duplicate'):
        study.checked_audits([audit,audit],plan,sources,engineering)
    tensor=Path(observations[0]['feature_file']);tensor.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='tensor missing or changed'):
        study.checked_audits([audit],plan,sources,engineering)
    tensor.unlink()
    with pytest.raises(FileNotFoundError):
        study.checked_audits([audit],plan,sources,engineering)
    cap.write_text('{"tampered":true}')
    with pytest.raises(ValueError,match='phase artifact changed'):
        study.checked_audits([audit],plan,sources,engineering)


def test_remaining_count_never_deducts_unverified_or_incomplete_sequences():
    plan={'fresh_reads':[{'sequence':'seq6','frame':1},{'sequence':'seq6','frame':2},
                         {'sequence':'seq8','frame':1}]}
    assert study.remaining_count(plan,{})==3
    assert study.remaining_count(plan,{'seq6':Path('/audited')})==1
