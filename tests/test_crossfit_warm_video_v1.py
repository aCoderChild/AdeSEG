"""CPU-only checks of the gated full warm-video adapter."""
import json
from types import SimpleNamespace
import pytest
from scripts import run_crossfit_warm_video_v1 as video
from scripts import run_crossfit_warm_video_study_v1 as study
from scripts.memory_token_io import digest


def test_full_warm_schedule_requires_separate_reference(tmp_path):
    tasks=study.phases('full',tmp_path,tmp_path/'gate.json','seq6',12)
    assert [x[0] for x in tasks]==['seq6_reference','seq6_capture','seq6_audit']
    assert '--reference' not in tasks[0][1] and '--reference' in tasks[1][1]
    assert '--certificate' in tasks[0][1] and '--certificate' in tasks[2][1]
    with pytest.raises(ValueError,match='short gate'):study.phases('full',tmp_path)


def test_missing_gate_fails_before_inference(tmp_path):
    from scripts.certify_crossfit_warm_gate_v1 import verify_certificate
    with pytest.raises(FileNotFoundError):verify_certificate(tmp_path/'certificate.json')


def test_video_inputs_use_viewport_coordinates(monkeypatch,tmp_path):
    import numpy as np
    prompt={'first_detection_index':0,'box_viewport_xyxy':[1,2,4,5],
            'points_viewport':[[2,3],[3,4]]}
    path=tmp_path/'prompt.json';path.write_text(json.dumps(prompt))
    warm={'fresh_prompts':{'seq23':{'prompt_metadata_path':str(path),'prompt_metadata_sha256':digest(path),
          'box_field':'box_viewport_xyxy','points_field':'points_viewport','roi_xyxy':[446,0,1280,720]}}}
    monkeypatch.setattr(video,'sequence_inputs',lambda plan,seq:([1,2],[],0,[1],np.ones((720,834),bool),None,None,None))
    result=video.video_inputs('seq23',{},warm)
    assert result[5]==[1,2,4,5] and result[6]==[[2,3],[3,4]] and result[7]==[446,0,1280,720]
    path.write_text('{}')
    with pytest.raises(ValueError,match='Prompt changed'):video.video_inputs('seq23',{},warm)


def test_per_sequence_audit_reservation_counts_written_bytes(monkeypatch,tmp_path):
    import shutil
    count=12;written=7*study.PER_READ;budget=count*study.PER_READ
    minimum=study.GUARD+study.MARGIN+budget-written
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=minimum))
    assert study.headroom_with_written(tmp_path,count,written,budget)['minimum_bytes']==minimum
    with pytest.raises(ValueError,match='Insufficient headroom'):
        study.headroom_with_written(tmp_path,count,0,budget)


def test_certified_seq6_reuse_requires_phase_and_intact_tensors(monkeypatch,tmp_path):
    monkeypatch.setattr(study,'BASE',tmp_path)
    folder=tmp_path/'gate/seq6/capture';folder.mkdir(parents=True)
    rows=[]
    for frame in (24,90):
        row={'frame':frame}
        for key in ('feature','snapshot'):
            path=folder/f'{key}{frame}.pt';path.write_bytes(f'{key}{frame}'.encode())
            row[key+'_file']=str(path);row[key+'_sha256']=digest(path)
        rows.append(row)
    receipt=folder/'receipt.json';receipt.write_text(json.dumps({'passed':True,'sequence':'seq6','mode':'capture',
        'warm_plan_sha256':study.WARM_SHA,'label_guard':{'open_attempts':0},'target_frames':[24,90],'observations':rows}))
    root=folder.parent.parent;log=root/'seq6_capture.log';log.write_text('done')
    phase={'phase':'seq6_capture','passed':True,'artifact_path':str(receipt),'artifact_sha256':digest(receipt),'log_sha256':digest(log)}
    (root/'phases.jsonl').write_text(json.dumps(phase)+'\n')
    ref=tmp_path/'historical.json';ref.write_text('{}')
    certificate={'seq6_gate_capture':{'receipt_path':str(receipt),'receipt_sha256':digest(receipt)},
                 'seq6_historical_reference':{'reference_path':str(ref),'reference_sha256':digest(ref)}}
    plan={'fresh_reads':[{'sequence':'seq6','frame':24},{'sequence':'seq6','frame':90}]}
    assert study.certified_seq6(certificate,plan)==root
    tensor=tmp_path/'gate/seq6/capture/feature24.pt';tensor.write_bytes(b'changed')
    with pytest.raises(ValueError,match='gate tensor changed'):study.certified_seq6(certificate,plan)
    tensor.unlink()
    with pytest.raises(FileNotFoundError):study.certified_seq6(certificate,plan)


def test_certificate_rejects_changed_gate_and_checkpoint(monkeypatch,tmp_path):
    import scripts.certify_crossfit_warm_gate_v1 as certify
    monkeypatch.setattr(certify,'BASE',tmp_path)
    existing=tmp_path/'existing';existing.mkdir();monkeypatch.setattr(certify,'EXISTING',existing)
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'checkpoint')
    manifest=existing/'manifest.json';manifest.write_text(json.dumps({'checkpoint_path':str(checkpoint)}))
    history={'reference_path':'old','reference_sha256':'sha'};gate_capture={'receipt_path':'capture','receipt_sha256':'hash'}
    warm={'existing_manifest_sha256':digest(manifest),'seq6_historical':history,
          'old_identity':{'checkpoint_sha256':digest(checkpoint)}}
    monkeypatch.setattr(certify,'warm_plan',lambda:warm)
    monkeypatch.setattr(certify,'verify_old_mps_hooks',lambda _:None)
    monkeypatch.setattr(certify,'sources',lambda:{'source':'hash'})
    monkeypatch.setattr(certify,'phase_artifact',lambda root,name,path:digest(path))
    gate=tmp_path/'gate/audit/audit.json';gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps({'passed':True,'warm_plan_sha256':certify.WARM_SHA,
        'label_guard':{'open_attempts':0},'runs':{'seq6_capture':gate_capture}}))
    folder=tmp_path/'certificate';folder.mkdir();path=folder/'certificate.json'
    record={'passed':True,'warm_plan_sha256':certify.WARM_SHA,'source_sha256':{'source':'hash'},
        'certifier_sha256':digest(certify.Path(certify.__file__)),'label_guard':{'open_attempts':0},
        'warm_gate_path':str(gate),'warm_gate_sha256':digest(gate),'seq6_gate_capture':gate_capture,
        'seq6_historical_reference':history,'existing_manifest_sha256':digest(manifest),
        'checkpoint_path':str(checkpoint),'checkpoint_sha256':digest(checkpoint)}
    path.write_text(json.dumps(record));(folder/'completion.json').write_text(json.dumps(
        {'passed':True,'certificate_sha256':digest(path),'label_guard':{'open_attempts':0}}))
    assert certify.verify_certificate(path)[0]['passed']
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):certify.verify_certificate(path)
    checkpoint.write_bytes(b'checkpoint');gate.write_text('{}')
    with pytest.raises(ValueError,match='gate changed'):certify.verify_certificate(path)
