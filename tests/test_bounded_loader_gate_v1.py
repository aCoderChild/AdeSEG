import json
from pathlib import Path

import pytest
import torch

from scripts import audit_bounded_loader_gate_v1 as audit
from scripts import run_bounded_loader_gate_study_v1 as launch
from scripts.memory_token_io import digest


def test_gate_audit_compares_fresh_process_tensors_and_reports_difference(tmp_path, monkeypatch):
    from scripts import certify_crossfit_warm_gate_v2 as certificate
    monkeypatch.setattr(audit, 'OUT', tmp_path)
    monkeypatch.setattr(launch, 'sources', lambda: {})
    monkeypatch.setattr(certificate, 'verify_certificate', lambda path: ({}, 'cert'))
    (tmp_path/'run_manifest.json').write_text(json.dumps({'source_sha256':{},
        'phases':['eager','bounded','audit'],'frames_staged':7,'certificate_path':'unused',
        'certificate_sha256':'cert','max_process_rss_bytes':6*1024**3,
        'min_disk_bytes':26*1024**3,'min_available_ram_bytes':4*1024**3,'cache_frames':4}))
    rows = []
    for mode, pid in [('eager', 11), ('bounded', 12)]:
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
            'certificate_sha256':'cert','checkpoint_sha256':'ck','predictor_sha256':'pred','misc_sha256':'misc',
            'access_contract':{},'frame_sha256':['a']*7,'model_state_sha256':'model',
            'conditioning_equals_archive':True,'conditioning_generated_difference_pixels':0,
            'rows':observations,'features':features,'positions':[],'selection':[],'anchor_events':[],'token_window_events':[],
            'peak_process_rss_bytes':123,'sampled_peak_mps_driver_bytes':456,
            'cache':None if mode=='eager' else {'final_sha256_verified':True,'peak_cached_tensor_bytes':4,
                                               'limit_bytes':8,'evictions':2}}
        receipt=folder/'receipt.json'; receipt.write_text(json.dumps(rec)); log=tmp_path/f'{mode}.log'; log.write_text('ok')
        rows.append({'phase':mode,'passed':True,'artifact_path':str(receipt),'artifact_sha256':digest(receipt),
                     'log_sha256':digest(log),'resource_monitor':{'minimum_sampled_disk_free_bytes':27*1024**3,
                     'minimum_sampled_system_available_ram_bytes':5*1024**3,'peak_sampled_process_rss_bytes':123}})
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    result=audit.audit(tmp_path); assert result['passed'] and len(result['comparisons'])==5
    bounded=tmp_path/'bounded/receipt.json'; cached=json.loads(bounded.read_text())
    cached['cache']['evictions']=0; bounded.write_text(json.dumps(cached)); rows[1]['artifact_sha256']=digest(bounded)
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='cache evidence differs'): audit.audit(tmp_path)
    cached['cache']['evictions']=2; bounded.write_text(json.dumps(cached)); rows[1]['artifact_sha256']=digest(bounded)
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    path=tmp_path/'bounded/000003.pt'; values=torch.load(path,weights_only=True); values['logits'][0,0,0,0]=-1
    torch.save(values,path)
    receipt=tmp_path/'bounded/receipt.json'; rec=json.loads(receipt.read_text())
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
