import json
from pathlib import Path

import pytest
import torch

from scripts import audit_mps_memory_diagnostic_v2 as audit
from scripts import run_mps_memory_diagnostic_study_v2 as launch
from scripts.memory_token_io import digest


def test_gate_audit_compares_fresh_process_tensors_and_reports_difference(tmp_path, monkeypatch):
    from scripts import certify_crossfit_warm_gate_v2 as certificate
    monkeypatch.setattr(audit, 'OUT', tmp_path)
    (tmp_path/'fake.py').write_text('original')
    source_map={'fake.py':digest(tmp_path/'fake.py')}
    monkeypatch.setattr(launch, 'sources', lambda: source_map)
    monkeypatch.setattr(launch, 'AUTHOR_FILES', ())
    snapshot_sha=launch.snapshot_sources(tmp_path/'source_snapshot', tmp_path, source_map)
    monkeypatch.setattr(certificate, 'verify_certificate', lambda path: ({}, 'cert'))
    (tmp_path/'run_manifest.json').write_text(json.dumps({'source_sha256':source_map,
        'phases':['observe','clear_idle_cache','audit'],'frames_staged':7,'certificate_path':'unused',
        'certificate_sha256':'cert','source_snapshot_manifest_sha256':snapshot_sha,'max_process_rss_bytes':6*1024**3,
        'min_disk_bytes':26*1024**3,'min_available_ram_bytes':4*1024**3,'cache_frames':4}))
    rows = []
    for mode, pid in [('observe', 11), ('clear_idle_cache', 12)]:
        folder = tmp_path/mode; folder.mkdir(); observations = []
        for frame in range(2,7):
            values = {name: torch.ones(1,1,2,2) for name in
                      ('logits','pred_masks','maskmem_features','obj_ptr','object_score_logits')}
            path=folder/f'{frame:06d}.pt'; torch.save(values,path)
            observations.append({'frame':frame,'tensor_path':str(path),'tensor_sha256':digest(path)})
        features=[]
        for frame in range(3,7):
            path=folder/f'features_{frame:06d}.pt'
            torch.save({'early':torch.ones(1,2,2,2),'deep':torch.ones(1,3,2,2)},path)
            features.append({'frame':frame,'path':str(path),'sha256':digest(path)})
        rec={'passed':True,'mode':mode,'sequence':'seq11','label_guard':{'open_attempts':0},'process_id':pid,
            'run_uuid':mode,'frames_staged':7,'prompt_frame':2,'roi_xyxy':[0,0,4,4],
            'certificate_sha256':'cert','min_available_ram_bytes':4*1024**3,'max_process_rss_bytes':6*1024**3,'checkpoint_sha256':'ck','predictor_sha256':'pred','wrapper_sha256':'wrapper','misc_sha256':'misc',
            'access_contract':{},'frame_sha256':['a']*7,'model_state_sha256':'model',
            'conditioning_equals_archive':True,'generated_equals_archived':True,'conditioning_generated_difference_pixels':0,
            'rows':observations,'features':features,'positions':[],'selection':[],'anchor_events':[],'token_window_events':[],
            'peak_process_rss_bytes':123,'sampled_peak_mps_driver_bytes':456,
            'cache':{'final_sha256_verified':True,'peak_cached_tensor_bytes':4,
                                               'limit_bytes':8,'evictions':2}}
        events=[]
        for frame in range(2,7):
            before={'system_available_bytes':5*1024**3,'process_rss_bytes':123,
                    'mps_live_tensor_bytes':500,'mps_driver_bytes':1000}
            after={**before, 'mps_driver_bytes':900 if mode=='clear_idle_cache' else 1000}
            events.append({'frame':frame,'mode':mode,'monotonic_seconds':float(frame),
                           'before':before,'after':after,
                           'driver_change_bytes':after['mps_driver_bytes']-before['mps_driver_bytes']})
        memory=folder/'memory.jsonl'; memory.write_text(''.join(json.dumps(x)+'\n' for x in events))
        rec['memory_trace_sha256']=digest(memory)
        milestones=folder/'milestones.jsonl'
        sample={'system_available_bytes':5*1024**3,'process_rss_bytes':123,
                'mps_live_tensor_bytes':500,'mps_driver_bytes':1000}
        names=['before_model_load','after_model_load','before_state_init','after_state_init',
               'before_prompt','after_prompt']
        milestones.write_text(''.join(json.dumps({'name':name,'monotonic_seconds':float(i),'sample':sample})+'\n'
                                      for i,name in enumerate(names)))
        rec['milestone_trace_sha256']=digest(milestones)
        receipt=folder/'receipt.json'; receipt.write_text(json.dumps(rec)); log=tmp_path/f'{mode}.log'; log.write_text('ok')
        rows.append({'phase':mode,'passed':True,'artifact_path':str(receipt),'artifact_sha256':digest(receipt),
                     'log_sha256':digest(log),'resource_monitor':{'minimum_sampled_disk_free_bytes':27*1024**3,
                     'minimum_sampled_system_available_ram_bytes':5*1024**3,'peak_sampled_process_rss_bytes':123}})
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    result=audit.audit(tmp_path); assert result['passed'] and len(result['comparisons'])==5
    milestone_file=tmp_path/'observe/milestones.jsonl'; original_milestones=milestone_file.read_text()
    events=[json.loads(x) for x in original_milestones.splitlines()]; events[0]['name']='wrong'
    milestone_file.write_text(''.join(json.dumps(x)+'\n' for x in events))
    milestone_receipt=tmp_path/'observe/receipt.json'; recorded=json.loads(milestone_receipt.read_text())
    recorded['milestone_trace_sha256']=digest(milestone_file); milestone_receipt.write_text(json.dumps(recorded))
    rows[0]['artifact_sha256']=digest(milestone_receipt)
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='Milestone trace coverage differs'): audit.audit(tmp_path)
    milestone_file.write_text(original_milestones); recorded['milestone_trace_sha256']=digest(milestone_file)
    milestone_receipt.write_text(json.dumps(recorded)); rows[0]['artifact_sha256']=digest(milestone_receipt)
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    snapshot_file=tmp_path/'source_snapshot/fake.py'; snapshot_file.write_text('tampered')
    with pytest.raises(ValueError,match='Diagnostic source snapshot changed'): audit.audit(tmp_path)
    snapshot_file.write_text('original')
    # Paired identical but invalid frame geometry must still be rejected.
    for field, bad in [('frames_staged',8), ('prompt_frame',3)]:
        saved=[]
        for index, mode in enumerate(('observe','clear_idle_cache')):
            receipt=tmp_path/mode/'receipt.json'; original_receipt=receipt.read_text(); saved.append(original_receipt)
            altered=json.loads(original_receipt); altered[field]=bad; receipt.write_text(json.dumps(altered))
            rows[index]['artifact_sha256']=digest(receipt)
        (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with pytest.raises(ValueError,match='frame coverage or conditioning'): audit.audit(tmp_path)
        for index, mode in enumerate(('observe','clear_idle_cache')):
            receipt=tmp_path/mode/'receipt.json'; receipt.write_text(saved[index]); rows[index]['artifact_sha256']=digest(receipt)
        (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    saved=[]
    for index, mode in enumerate(('observe','clear_idle_cache')):
        receipt=tmp_path/mode/'receipt.json'; original_receipt=receipt.read_text(); saved.append(original_receipt)
        altered=json.loads(original_receipt); altered['rows'][1]['frame']=2; receipt.write_text(json.dumps(altered))
        rows[index]['artifact_sha256']=digest(receipt)
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='frame coverage or conditioning'): audit.audit(tmp_path)
    for index, mode in enumerate(('observe','clear_idle_cache')):
        receipt=tmp_path/mode/'receipt.json'; receipt.write_text(saved[index]); rows[index]['artifact_sha256']=digest(receipt)
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    trace=tmp_path/'clear_idle_cache/memory.jsonl'
    original=trace.read_text(); samples=[json.loads(line) for line in original.splitlines()]
    samples[0]['driver_change_bytes']=17
    trace.write_text(''.join(json.dumps(x)+'\n' for x in samples))
    receipt=tmp_path/'clear_idle_cache/receipt.json'; changed=json.loads(receipt.read_text())
    changed['memory_trace_sha256']=digest(trace); receipt.write_text(json.dumps(changed))
    rows[1]['artifact_sha256']=digest(receipt)
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='Memory driver delta differs'): audit.audit(tmp_path)
    trace.write_text(original); changed['memory_trace_sha256']=digest(trace); receipt.write_text(json.dumps(changed))
    rows[1]['artifact_sha256']=digest(receipt)
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    audit_folder=tmp_path/'audit'; audit_folder.mkdir()
    audit_file=audit_folder/'audit.json'; audit_file.write_text(json.dumps(result))
    audit_log=tmp_path/'audit.log'; audit_log.write_text('ok')
    audit_row={'phase':'audit','passed':True,'artifact_path':str(audit_file),
               'artifact_sha256':digest(audit_file),'log_sha256':digest(audit_log)}
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows+[audit_row]))
    assert audit.audit(tmp_path)['passed']
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    # A mask bit can flip around zero while same-device logits are within
    # tolerance; report it without requiring bit identity.
    for index,mode in enumerate(('observe','clear_idle_cache')):
        path=tmp_path/mode/'000002.pt'; values=torch.load(path,weights_only=True)
        values['logits'][0,0,0,0]=1e-8 if mode=='observe' else -1e-8; torch.save(values,path)
        recpath=tmp_path/mode/'receipt.json'; data=json.loads(recpath.read_text())
        data['rows'][0]['tensor_sha256']=digest(path); recpath.write_text(json.dumps(data))
        rows[index]['artifact_sha256']=digest(recpath)
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    result=audit.audit(tmp_path)
    assert result['passed'] and result['comparisons'][0]['tensors']['logits']['binary_mask_differing_pixels']==1
    assert result['comparisons'][0]['tensors']['logits']['sign_flip_max_abs_logit'] == pytest.approx(1e-8)
    assert result['comparisons'][0]['tensors']['logits']['sign_flip_max_abs_difference'] == pytest.approx(2e-8)
    bounded=tmp_path/'clear_idle_cache/receipt.json'; cached=json.loads(bounded.read_text())
    cached['cache']['evictions']=0; bounded.write_text(json.dumps(cached)); rows[1]['artifact_sha256']=digest(bounded)
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='cache evidence differs'): audit.audit(tmp_path)
    cached['cache']['evictions']=2; bounded.write_text(json.dumps(cached)); rows[1]['artifact_sha256']=digest(bounded)
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    path=tmp_path/'clear_idle_cache/000003.pt'; values=torch.load(path,weights_only=True); values['logits'][0,0,0,0]=-1
    torch.save(values,path)
    receipt=tmp_path/'clear_idle_cache/receipt.json'; rec=json.loads(receipt.read_text())
    rec['rows'][1]['tensor_sha256']=digest(path); receipt.write_text(json.dumps(rec))
    rows[1]['artifact_sha256']=digest(receipt); (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    result=audit.audit(tmp_path); assert not result['passed']
    assert {'frame':3,'tensor':'logits'} in result['errors']
    assert result['comparisons'][1]['tensors']['logits']['binary_mask_differing_pixels']==1
    manifest=tmp_path/'run_manifest.json'; data=json.loads(manifest.read_text()); data['min_disk_bytes']=1
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='resource limits'): audit.audit(tmp_path)
    data['min_disk_bytes']=26*1024**3; manifest.write_text(json.dumps(data))
    del rec['peak_process_rss_bytes']; receipt.write_text(json.dumps(rec))
    rows[1]['artifact_sha256']=digest(receipt); (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='Diagnostic receipt differs'): audit.audit(tmp_path)


def test_resource_gate_rejects_low_headroom(tmp_path, monkeypatch):
    class V: available=100
    class D: free=100
    monkeypatch.setattr(launch.psutil,'virtual_memory',lambda:V())
    monkeypatch.setattr(launch.shutil,'disk_usage',lambda root:D())
    with pytest.raises(ValueError,match='Insufficient resources'):
        launch.resources(tmp_path,101,50)
    with pytest.raises(ValueError,match='Insufficient resources'):
        launch.resources(tmp_path,50,101)
    assert launch.resources(tmp_path,50,50)==(100,100)


def test_minimal_feature_observer_preserves_return_and_restores(tmp_path):
    from modeling.bounded_loader_feature_observer_v1 import observe_decoder_features
    class Model:
        def _get_image_feature(self, state, frame_idx, batch_size): return ('original-feature-result',)
        def _forward_sam_heads(self, **kwargs): return ('original-head-result',)
    model=Model(); records=[]
    with observe_decoder_features(model,{3},tmp_path,records):
        assert model._get_image_feature({},3,1)==('original-feature-result',)
        result=model._forward_sam_heads(backbone_features=torch.ones(1,256,64,64),
                                      high_res_features=[torch.ones(1,32,256,256)])
        assert result==('original-head-result',)
    assert '_get_image_feature' not in vars(model) and '_forward_sam_heads' not in vars(model)
    assert records[0]['frame']==3 and digest(records[0]['path'])==records[0]['sha256']
    with pytest.raises(RuntimeError,match='failure'):
        with observe_decoder_features(model,{3},tmp_path,[]):
            raise RuntimeError('failure')
    assert '_get_image_feature' not in vars(model) and '_forward_sam_heads' not in vars(model)


def test_memory_sampling_only_clears_requested_mode(tmp_path, monkeypatch):
    from scripts import run_mps_memory_diagnostic_v2 as worker
    calls=[]
    monkeypatch.setattr(worker.torch.mps, 'synchronize', lambda: calls.append('sync'))
    monkeypatch.setattr(worker.torch.mps, 'empty_cache', lambda: calls.append('clear'))
    snapshots=iter([{'system_available_bytes':5, 'process_rss_bytes':2, 'mps_live_tensor_bytes':1, 'mps_driver_bytes':10},
                    {'system_available_bytes':5, 'process_rss_bytes':2, 'mps_live_tensor_bytes':1, 'mps_driver_bytes':10},
                    {'system_available_bytes':5, 'process_rss_bytes':2, 'mps_live_tensor_bytes':1, 'mps_driver_bytes':10},
                    {'system_available_bytes':6, 'process_rss_bytes':2, 'mps_live_tensor_bytes':1, 'mps_driver_bytes':7}])
    monkeypatch.setattr(worker, 'memory_snapshot', lambda: next(snapshots))
    path=tmp_path/'memory.jsonl'
    worker.record_memory(path, 2, 'observe', 4, 10)
    worker.record_memory(path, 3, 'clear_idle_cache', 4, 10)
    assert calls == ['sync','sync','sync','clear','sync']
    events=[json.loads(x) for x in path.read_text().splitlines()]
    assert [x['driver_change_bytes'] for x in events] == [0, -3]
    with pytest.raises(ValueError, match='Unknown memory diagnostic mode'):
        worker.record_memory(path, 4, 'wrong', 4, 10)


def test_observed_resource_violation_stops_and_preserves_sample(tmp_path, monkeypatch):
    from scripts import run_mps_memory_diagnostic_v2 as worker
    calls=[]
    monkeypatch.setattr(worker.torch.mps, 'synchronize', lambda: calls.append('sync'))
    monkeypatch.setattr(worker.torch.mps, 'empty_cache', lambda: calls.append('clear'))
    monkeypatch.setattr(worker, 'memory_snapshot', lambda: {'system_available_bytes':3, 'process_rss_bytes':2,
        'mps_live_tensor_bytes':1, 'mps_driver_bytes':10})
    path=tmp_path/'memory.jsonl'
    with pytest.raises(RuntimeError, match='resource limit before'):
        worker.record_memory(path, 2, 'clear_idle_cache', 4, 10)
    assert calls == ['sync']
    assert json.loads(path.read_text())['stopped_at'] == 'before'
    samples=iter([{'system_available_bytes':5, 'process_rss_bytes':2, 'mps_live_tensor_bytes':1, 'mps_driver_bytes':10},
                  {'system_available_bytes':3, 'process_rss_bytes':2, 'mps_live_tensor_bytes':1, 'mps_driver_bytes':8}])
    monkeypatch.setattr(worker, 'memory_snapshot', lambda: next(samples))
    with pytest.raises(RuntimeError, match='resource limit after'):
        worker.record_memory(path, 3, 'clear_idle_cache', 4, 10)
    assert json.loads(path.read_text().splitlines()[-1])['stopped_at'] == 'after'


def test_snapshot_detects_source_mutation(tmp_path):
    source=tmp_path/'source'; source.mkdir(); (source/'code.py').write_text('original')
    expected={'code.py':digest(source/'code.py')}
    manifest_sha=launch.snapshot_sources(tmp_path/'snapshot',source,expected)
    assert digest(tmp_path/'snapshot/snapshot_manifest.json') == manifest_sha
    assert digest(tmp_path/'snapshot/code.py') == expected['code.py']
    (source/'code.py').write_text('modified')
    with pytest.raises(ValueError,match='Source changed before snapshot'):
        launch.snapshot_sources(tmp_path/'second',source,expected)


def test_milestone_stops_and_preserves_resource_violation(tmp_path,monkeypatch):
    from scripts import run_mps_memory_diagnostic_v2 as worker
    monkeypatch.setattr(worker.torch.mps,'synchronize',lambda:None)
    sample={'system_available_bytes':3,'process_rss_bytes':2,'mps_live_tensor_bytes':1,'mps_driver_bytes':7}
    monkeypatch.setattr(worker,'memory_snapshot',lambda:sample)
    path=tmp_path/'milestones.jsonl'
    with pytest.raises(ValueError,match='milestone observed resource limit'):
        worker.record_milestone(path,'after_model_load',4,10)
    event=json.loads(path.read_text())
    assert event['name']=='after_model_load' and event['sample']==sample and event['stopped_on_resource_limit'] is True


def test_actual_v2_source_closure_binds_own_entrypoints():
    actual=launch.sources()
    for name in ('scripts/run_mps_memory_diagnostic_study_v2.py',
                 'scripts/run_mps_memory_diagnostic_v2.py',
                 'scripts/audit_mps_memory_diagnostic_v2.py',
                 'protocols/MPS_MEMORY_DIAGNOSTIC_V2.md'):
        assert actual[name] == digest(launch.ROOT/name)
