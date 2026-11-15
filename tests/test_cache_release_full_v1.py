import numpy as np
from PIL import Image
import pytest
import torch

from scripts.audit_cache_release_full_v1 import compare_masks, compare_tree
from scripts.memory_token_io import digest
from scripts.run_cache_release_full_v1 import run_capture


def test_tensor_tree_uses_float_tolerance_but_exact_nonfloat():
    left={'early':torch.tensor([1.0]),'index':torch.tensor([2]),'meta':('ok',None)}
    near={'early':torch.tensor([1.000001]),'index':torch.tensor([2]),'meta':('ok',None)}
    assert all(x['within_tolerance'] for x in compare_tree(left,near))
    near['early']=torch.tensor([1.1]); near['index']=torch.tensor([3])
    failures=[x['path'] for x in compare_tree(left,near) if not x['within_tolerance']]
    assert failures==['root.early','root.index']


def test_full_mask_agreement_reports_pixel_differences(tmp_path):
    a=tmp_path/'a'; b=tmp_path/'b'; a.mkdir(); b.mkdir()
    x=np.zeros((10,10),np.uint8); x[2:8,2:8]=255; y=x.copy(); y[2,2]=0
    pa=a/'000000.png'; pb=b/'000000.png'; Image.fromarray(x).save(pa); Image.fromarray(y).save(pb)
    left={'rows':[{'frame':0,'mask_sha256':digest(pa)}]}; right={'rows':[{'frame':0,'mask_sha256':digest(pb)}]}
    result=compare_masks(left,a,right,b)
    assert result[0]['differing_pixels']==1 and result[0]['mask_dice']==pytest.approx(70/71)
    pb.write_bytes(b'tamper')
    with pytest.raises(ValueError,match='Mask changed'): compare_masks(left,a,right,b)


def test_full_capture_rejects_invalid_configuration_before_sources(tmp_path):
    with pytest.raises(ValueError,match='Invalid bounded capture configuration'):
        run_capture(sequence='seq11',mode='capture',output=tmp_path/'out',certificate_path='missing',
                    short_gate_path='missing',reference=None)


def test_full_capture_study_plan_precedes_audit_and_seq6_only_is_enabled(tmp_path):
    from scripts import run_cache_release_full_study_v1 as study
    tasks=study.phases(tmp_path,'seq6',tmp_path/'cert',tmp_path/'short',4)
    assert [x[0] for x in tasks]==['seq6_reference','seq6_capture','seq6_audit']
    assert tasks[1][1][tasks[1][1].index('--reference')+1]==str(tmp_path/'seq6/reference')
    assert study.ENABLE_REVIEWED_EXECUTION is True


def test_full_seq6_audit_binds_and_compares_eager_observations(tmp_path, monkeypatch):
    import json
    from scripts import audit_cache_release_full_v1 as full
    monkeypatch.setattr(full,'OUT',tmp_path)
    monkeypatch.setattr(full,'sources',lambda:{})
    monkeypatch.setattr(full,'warm_plan',lambda:{})
    monkeypatch.setattr(full,'validate_plan',lambda:{'fresh_reads':[{'sequence':'seq6'},{'sequence':'seq6'}]})
    monkeypatch.setattr(full,'verify_gate',lambda path:({'certificate_sha256':'cert'},'gate'))
    monkeypatch.setattr(full,'phase_artifact',lambda root,name,path:digest(path))
    phase_rows=[]; runs={}
    for mode,pid in [('reference',1),('capture',2),('historical',3),('oldcapture',4)]:
        folder=tmp_path/'seq6'/mode;folder.mkdir(parents=True)
        rows=[];observations=[]
        for frame in (24,90):
            mask=folder/f'{frame:06d}.png';Image.fromarray(np.zeros((3,3),np.uint8)).save(mask)
            rows.append({'frame':frame,'mask_sha256':digest(mask)})
            item={'frame':frame}
            for kind in ('feature','snapshot'):
                file=folder/f'{kind}_{frame}.pt';torch.save({'value':torch.tensor([1.0])},file)
                item[kind+'_file']=str(file);item[kind+'_sha256']=digest(file)
            observations.append(item)
        run={'rows':rows,'observations':observations,'positions':[],'selection':[],
             'anchor_events':[],'token_window_events':[],'process_id':pid,'run_uuid':mode}
        for name,field in (('conditioning_mask.png','conditioning_sha256'),('generated_mask.png','generated_sha256')):
            file=folder/name;Image.fromarray(np.zeros((3,3),np.uint8)).save(file);run[field]=digest(file)
        for name,field in (('prompt_early.pt','prompt_early_sha256'),('prototypes.pt','prototypes_sha256')):
            file=folder/name;torch.save({'value':torch.tensor([1.])},file);run[field]=digest(file)
        if mode in ('reference','capture'):
            sample={'system_available_bytes':5*1024**3,'process_rss_bytes':123,
                    'mps_live_tensor_bytes':500,'mps_driver_bytes':1000}
            setup=tmp_path/'seq6'/(mode+'_setup_cache.jsonl')
            setup.write_text(''.join(json.dumps({'name':name,'mode':'clear_idle_cache','before':sample,'after':sample,
                                                 'driver_change_bytes':0})+'\n'
                            for name in ('after_model_load','after_state_init','after_prompt')))
            framefile=tmp_path/'seq6'/(mode+'_frame_cache.jsonl')
            framefile.write_text(''.join(json.dumps({'frame':x['frame'],'mode':'clear_idle_cache','before':sample,
                                      'after':sample,'driver_change_bytes':0})+'\n' for x in rows))
            run['cache_setup_sha256']=digest(setup);run['cache_frame_sha256']=digest(framefile)
        run['bounded_base_source_sha256']=digest(full.ROOT/'scripts/run_cache_release_warm_base_v1.py')
        if mode=='capture': run['reference_receipt_sha256']=digest(tmp_path/'seq6/reference/receipt.json')
        receipt=folder/'receipt.json';receipt.write_text(json.dumps(run));runs[mode]=run
        if mode in ('reference','capture'):
            side=folder/'bounded_receipt.json'
            side.write_text(json.dumps({'passed':True,'sequence':'seq6','mode':mode,
                'original_receipt_sha256':digest(receipt),'certificate_sha256':'cert','short_gate_sha256':'gate',
                'source_sha256':{},'label_guard':{'open_attempts':0},'peak_process_rss_bytes':123,
                'cache':{'final_sha256_verified':True,'evictions':1},'process_id':pid,'run_uuid':mode}))
            phase_rows.append({'phase':f'seq6_{mode}','bounded_receipt_sha256':digest(side),
                'resource_monitor':{'minimum_sampled_disk_free_bytes':27*1024**3,
                  'minimum_sampled_system_available_ram_bytes':5*1024**3,'peak_sampled_process_rss_bytes':123,
                  'minimum_sampled_payload_margin_bytes':1,'planned_payload_bytes':2*full.PER_READ,
                  'written_payload_bytes':0}})
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in phase_rows))
    certfile=tmp_path/'cert';certfile.write_text('{}');gatefile=tmp_path/'gate';gatefile.write_text('{}')
    import shutil
    snapshot=tmp_path/'source_snapshot';snapshot.mkdir();snapshot_map={}
    for name in ('external/Lean-SAM2-token-window/sam2/sam2_video_predictor_swr_wrapper.py',
                 'external/Lean-SAM2-token-window/sam2/sam2_video_predictor_topk_insurance.py',
                 'external/Lean-SAM2-token-window/sam2/utils/misc.py'):
        source=full.ROOT/name;target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);snapshot_map[name]=digest(source)
    snapshot_manifest=snapshot/'snapshot_manifest.json'
    snapshot_manifest.write_text(json.dumps({'source_sha256':snapshot_map}))
    manifest={'sequence':'seq6','certificate_path':str(certfile),'certificate_sha256':'cert',
        'short_gate_path':str(gatefile),'short_gate_sha256':'gate','source_sha256':{},
        'memory_gate_path':'unused','memory_gate_sha256':'memory',
        'phases':['seq6_reference','seq6_capture','seq6_audit'],'cache_frames':4,
        'min_disk_bytes':26*1024**3,'min_available_ram_bytes':4*1024**3,'max_process_rss_bytes':6*1024**3,
        'planned_capture_reads':2,'planned_payload_bytes':2*full.PER_READ,
        'source_snapshot_manifest_sha256':digest(snapshot_manifest)}
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest))
    oldref=tmp_path/'seq6/historical/receipt.json';oldcap=tmp_path/'seq6/oldcapture/receipt.json'
    certificate={'warm_gate_sha256':'warm','seq6_historical_reference':{'reference_path':str(oldref),
        'reference_sha256':digest(oldref)},'seq6_gate_capture':{'receipt_path':str(oldcap),'receipt_sha256':digest(oldcap)}}
    monkeypatch.setattr(full,'verify_certificate',lambda path:(certificate,'cert'))
    from scripts import verify_mps_memory_diagnostic_v3 as memory_verifier
    monkeypatch.setattr(memory_verifier,'verify_gate',lambda path:({'certificate_sha256':'cert'},'memory'))
    monkeypatch.setattr(full,'checked_run',lambda root,seq,mode,*a:(runs[mode],digest(root/seq/mode/'receipt.json')))
    result=full.audit(tmp_path,'seq6',certfile,gatefile)
    assert result['passed'] and len(result['historical_observations'])==4
    assert result['minimum_observed_mask_dice']==1 and result['historical_traces']['selection']
    checks=iter([({'certificate_sha256':'cert'},'memory'),({'certificate_sha256':'cert'},'altered')])
    monkeypatch.setattr(memory_verifier,'verify_gate',lambda path:next(checks))
    with pytest.raises(ValueError,match='MPS memory diagnostic changed during audit'):
        full.audit(tmp_path,'seq6',certfile,gatefile)
    monkeypatch.setattr(memory_verifier,'verify_gate',lambda path:({'certificate_sha256':'cert'},'memory'))
    monkeypatch.setattr(memory_verifier,'verify_gate',lambda path:({'certificate_sha256':'cert'},'wrong'))
    with pytest.raises(ValueError,match='MPS memory diagnostic proof differs'):
        full.audit(tmp_path,'seq6',certfile,gatefile)
    monkeypatch.setattr(memory_verifier,'verify_gate',lambda path:({'certificate_sha256':'cert'},'memory'))
    trace=tmp_path/'seq6/capture_frame_cache.jsonl';original=trace.read_text()
    records=[json.loads(x) for x in original.splitlines()];records[0]['driver_change_bytes']=1
    trace.write_text(''.join(json.dumps(x)+'\n' for x in records))
    runs['capture']['cache_frame_sha256']=digest(trace)
    with pytest.raises(ValueError,match='Cache-release delta differs'):
        full.audit(tmp_path,'seq6',certfile,gatefile)
    trace.write_text(original);runs['capture']['cache_frame_sha256']=digest(trace)
    snapshot_file=snapshot/'external/Lean-SAM2-token-window/sam2/utils/misc.py'
    snapshot_file.write_bytes(b'tamper')
    with pytest.raises(ValueError,match='source snapshot changed'):
        full.audit(tmp_path,'seq6',certfile,gatefile)


def test_prompt_feature_drift_and_mask_tamper_are_reported(tmp_path):
    from scripts.audit_cache_release_full_v1 import compare_prompt_artifacts
    records=[]
    for name,value in [('a',1.),('b',1.1)]:
        folder=tmp_path/name;folder.mkdir();record={}
        for filename,field in (('prompt_early.pt','prompt_early_sha256'),('prototypes.pt','prototypes_sha256')):
            path=folder/filename;torch.save({'value':torch.tensor([value])},path);record[field]=digest(path)
        for filename,field in (('conditioning_mask.png','conditioning_sha256'),('generated_mask.png','generated_sha256')):
            path=folder/filename;Image.fromarray(np.zeros((4,4),np.uint8)).save(path);record[field]=digest(path)
        records.append(record)
    result=compare_prompt_artifacts(records[0],tmp_path/'a',records[1],tmp_path/'b')
    assert not result['tensors']['prompt_early.pt']['all_within_tolerance']
    assert result['masks']['conditioning_mask.png']['differing_pixels']==0
    (tmp_path/'b/generated_mask.png').write_bytes(b'tamper')
    with pytest.raises(ValueError,match='Prompt mask changed'):
        compare_prompt_artifacts(records[0],tmp_path/'a',records[1],tmp_path/'b')


def test_dynamic_payload_reserve_and_low_headroom(tmp_path, monkeypatch):
    from scripts import run_cache_release_full_study_v1 as study
    base=26*study.GIB; budget=1000
    assert study.required_disk(base,budget,0)==base+1000
    assert study.required_disk(base,budget,600)==base+400
    assert study.required_disk(base,budget,9000)==base
    class Usage: free=base+999
    class Memory: available=5*study.GIB
    monkeypatch.setattr(study.shutil,'disk_usage',lambda root:Usage())
    monkeypatch.setattr(study.psutil,'virtual_memory',lambda:Memory())
    with pytest.raises(ValueError,match='resources below guard'):
        study.run_monitored(['unused'],tmp_path/'log',tmp_path,tmp_path/'capture',base,4*study.GIB,6*study.GIB,budget)
    assert not (tmp_path/'log').exists()


def test_seq6_proof_fails_closed_when_missing_or_wrong(tmp_path,monkeypatch):
    import json
    from scripts import run_cache_release_full_study_v1 as study
    from scripts import audit_cache_release_full_v1 as full
    from scripts import crossfit_observer_capture_shared_v1 as shared
    monkeypatch.setattr(study,'OUT',tmp_path)
    with pytest.raises(FileNotFoundError): study.verify_seq6_proof(tmp_path/'missing','cert','gate')
    folder=tmp_path/'seq6/audit';folder.mkdir(parents=True);path=folder/'audit.json'
    proof={'passed':True,'sequence':'seq6','certificate_sha256':'cert','short_gate_sha256':'gate',
           'label_guard':{'open_attempts':0}}
    path.write_text(json.dumps(proof));(tmp_path/'run_manifest.json').write_text(json.dumps(
        {'certificate_path':'c','short_gate_path':'g'}))
    monkeypatch.setattr(shared,'phase_artifact',lambda root,name,path:digest(path))
    monkeypatch.setattr(full,'audit',lambda *a:{k:v for k,v in proof.items() if k!='label_guard'})
    assert study.verify_seq6_proof(path,'cert','gate')[1]==digest(path)
    with pytest.raises(ValueError,match='proof differs'):study.verify_seq6_proof(path,'wrong','gate')


def test_wrapper_calls_original_runner_and_restores_builder(tmp_path,monkeypatch):
    import json,sys
    from contextlib import contextmanager
    from types import SimpleNamespace
    from scripts import run_cache_release_full_v1 as wrapper
    monkeypatch.setattr(wrapper,'OUT',tmp_path)
    monkeypatch.setattr(wrapper,'validate_supervision',lambda **kwargs:None)
    monkeypatch.setattr(wrapper,'install_mask_open_guard',lambda data:{'open_attempts':0})
    monkeypatch.setattr(wrapper,'warm_plan',lambda:{'old_identity':{'fork_patches':{
        'sam2/sam2_video_predictor_topk_insurance.py':'predictor'}}})
    monkeypatch.setattr(wrapper,'validate_plan',lambda:{})
    monkeypatch.setattr(wrapper,'verify_certificate',lambda path:({},'cert'))
    gatepath=tmp_path/'gate/audit/audit.json';gatepath.parent.mkdir(parents=True);gatepath.write_text('{}')
    loaderfile=wrapper.ROOT/'modeling/bounded_sam2_video_loader_v2.py'
    (tmp_path/'gate/run_manifest.json').write_text(json.dumps({'source_sha256':{
        'modeling/bounded_sam2_video_loader_v2.py':digest(loaderfile)}}))
    monkeypatch.setattr(wrapper,'verify_gate',lambda path:({'certificate_sha256':'cert'},'gate'))
    fake_frame=SimpleNamespace(image_path=tmp_path/'image');fake_frame.image_path.write_bytes(b'image')
    monkeypatch.setattr(wrapper,'video_inputs',lambda *args:([fake_frame],[{'image_sha256':digest(fake_frame.image_path)}],
                                                            0,[],None,None,None,None))
    monkeypatch.setattr(wrapper,'sources',lambda:{})
    monkeypatch.setattr(wrapper,'resolve_swr_init_chain',lambda *args,**kwargs:None)
    sys.path.insert(0,str(wrapper.FORK))
    class FakeLoader:
        def verify_sources(self):self.verified=True
        def stats(self):return {'final_sha256_verified':True,'evictions':1}
    loader=FakeLoader()
    @contextmanager
    def install(*args,**kwargs):yield [loader]
    monkeypatch.setattr(wrapper,'install_bounded_eager_loader',install)
    class Model:
        image_size=32
        def init_state(self,video_path,**kwargs):return {'images':loader}
        def propagate_in_video(self,*args,**kwargs):return iter(())
    original_builder=lambda args,repo:Model()
    monkeypatch.setattr(wrapper.original,'build_author_predictor',original_builder)
    out=tmp_path/'output'
    def fake_main():
        model=wrapper.original.build_author_predictor(None,None)
        assert model.init_state(video_path=str(tmp_path))['images'] is loader
        import os,json
        out.mkdir();(out/'receipt.json').write_text(json.dumps({'process_id':os.getpid(),
            'run_uuid':'base-test-uuid','bounded_base_source_sha256':
                digest(wrapper.ROOT/'scripts/run_cache_release_warm_base_v1.py')}))
    monkeypatch.setattr(wrapper.original,'main',fake_main)
    cert=tmp_path/'cert';cert.write_text('{}')
    rec=wrapper.run_capture(sequence='seq6',mode='reference',output=out,certificate_path=cert,
                            short_gate_path=gatepath,cache_frames=4)
    assert rec['passed'] and rec['cache']['final_sha256_verified'] and loader.verified
    assert rec['run_uuid']=='base-test-uuid'
    assert wrapper.original.build_author_predictor is original_builder
    assert (out/'bounded_receipt.json').exists()


def test_bounded_base_source_identity_cannot_equal_frozen_eager_map():
    from scripts import run_crossfit_warm_video_v2 as eager
    from scripts import run_cache_release_warm_base_v1 as bounded
    eager_map={p:digest(eager.ROOT/p) for p in eager.SOURCES}
    bounded_map=bounded.bounded_base_source_hashes()
    assert bounded_map != eager_map
    assert bounded_map['scripts/run_cache_release_warm_base_v1.py'] == \
           digest(bounded.ROOT/'scripts/run_cache_release_warm_base_v1.py')


def test_worker_rejects_seq11_even_when_seq6_enabled(tmp_path,monkeypatch):
    from scripts import run_cache_release_full_v1 as worker
    monkeypatch.setattr(worker,'install_mask_open_guard',lambda _: {'open_attempts':0})
    monkeypatch.setattr(worker.sys,'argv',['worker','--sequence','seq11','--mode','reference',
        '--output',str(tmp_path/'out'),'--certificate',str(tmp_path/'cert'),
        '--short-gate',str(tmp_path/'gate'),'--batch-root',str(tmp_path)])
    with pytest.raises(ValueError,match='Only reviewed seq6'):worker.main()


def test_cache_release_source_closure_binds_schedule_and_proofs():
    from scripts import run_cache_release_full_v1 as worker
    sources=worker.sources()
    for name in ('scripts/run_cache_release_full_v1.py','scripts/run_cache_release_full_study_v1.py',
                 'scripts/audit_cache_release_full_v1.py','scripts/run_cache_release_warm_base_v1.py',
                 'scripts/run_mps_memory_diagnostic_v3.py','scripts/verify_mps_memory_diagnostic_v3.py',
                 'protocols/CACHE_RELEASE_FULL_SEQ6_V1.md'):
        assert sources[name] == digest(worker.ROOT/name)


def test_direct_worker_and_base_cli_fail_when_not_reviewed(tmp_path,monkeypatch):
    import sys
    from scripts import run_cache_release_full_v1 as worker
    from scripts import run_cache_release_warm_base_v1 as base
    monkeypatch.setattr(worker,'ENABLE_REVIEWED_EXECUTION',False)
    with pytest.raises(ValueError,match='disabled pending review'):
        worker.run_capture(sequence='seq6',mode='reference',output=tmp_path/'out',
             certificate_path=tmp_path/'cert',short_gate_path=tmp_path/'gate')
    monkeypatch.setattr(sys,'argv',['base','--sequence','seq6','--mode','reference',
        '--output',str(tmp_path/'out'),'--certificate',str(tmp_path/'cert'),
        '--batch-root',str(tmp_path),'--short-gate',str(tmp_path/'gate'),'--cache-frames','4'])
    with pytest.raises(ValueError,match='disabled pending review'):base.main()


def test_supervised_context_rejects_wrong_phase_and_proof(tmp_path,monkeypatch):
    import json
    from scripts import run_cache_release_full_v1 as worker
    from scripts import verify_mps_memory_diagnostic_v3 as memory_verifier
    monkeypatch.setattr(worker,'ENABLE_REVIEWED_EXECUTION',True)
    monkeypatch.setattr(worker,'OUT',tmp_path)
    monkeypatch.setattr(worker,'sources',lambda:{})
    monkeypatch.setattr(worker,'assert_live_supervisor',lambda manifest:None)
    monkeypatch.setattr(worker,'verify_certificate',lambda path:({},'cert-sha'))
    monkeypatch.setattr(worker,'verify_gate',lambda path:({},'gate-sha'))
    monkeypatch.setattr(memory_verifier,'verify_gate',lambda path:({'certificate_sha256':'cert-sha'},'memory-sha'))
    batch=tmp_path/'batch'; batch.mkdir(); cert=tmp_path/'cert'; gate=tmp_path/'gate'
    cert.write_text('{}'); gate.write_text('{}')
    manifest={'sequence':'seq6','certificate_path':str(cert),'short_gate_path':str(gate),'cache_frames':4,
              'source_sha256':{},'certificate_sha256':'cert-sha','short_gate_sha256':'gate-sha',
              'memory_gate_path':'memory','memory_gate_sha256':'memory-sha'}
    (batch/'run_manifest.json').write_text(json.dumps(manifest))
    args=dict(sequence='seq6',mode='reference',output=batch/'seq6/reference',certificate_path=cert,
              short_gate_path=gate,reference=None,cache_frames=4)
    worker.validate_supervision(**args)
    (batch/'phases.jsonl').write_text(json.dumps({'phase':'seq6_reference'})+'\n')
    with pytest.raises(ValueError,match='supervised phase differs'):worker.validate_supervision(**args)
    (batch/'phases.jsonl').write_text('')
    manifest['memory_gate_sha256']='wrong';(batch/'run_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='supervised proof differs'):worker.validate_supervision(**args)


def test_live_supervisor_requires_direct_parent_script_and_held_lock(tmp_path,monkeypatch):
    from scripts import run_cache_release_full_v1 as worker
    monkeypatch.setattr(worker,'OUT',tmp_path)
    lock=tmp_path/'.source_group_capture_resume_v1.lock';lock.write_text('')
    class Process:
        def is_running(self):return True
        def create_time(self):return 123.
        def cmdline(self):return ['python',str(worker.ROOT/'scripts/run_cache_release_full_study_v1.py')]
        def cwd(self):return str(worker.ROOT)
    monkeypatch.setattr(worker.os,'getppid',lambda:42)
    monkeypatch.setattr(worker.psutil,'Process',lambda pid:Process())
    manifest={'supervisor_pid':42,'supervisor_create_time':123.}
    with pytest.raises(ValueError,match='lock is not held'):worker.assert_live_supervisor(manifest)
    # A real independently opened descriptor must observe the supervisor lock.
    with lock.open('a') as held:
        worker.fcntl.flock(held.fileno(),worker.fcntl.LOCK_EX|worker.fcntl.LOCK_NB)
        try: worker.assert_live_supervisor(manifest)
        finally: worker.fcntl.flock(held.fileno(),worker.fcntl.LOCK_UN)
    def blocked(fd,flags):raise BlockingIOError()
    monkeypatch.setattr(worker.fcntl,'flock',blocked)
    worker.assert_live_supervisor(manifest)
    with pytest.raises(ValueError,match='parent differs'):
        worker.assert_live_supervisor({'supervisor_pid':43,'supervisor_create_time':123.})
    monkeypatch.setattr(Process,'cmdline',lambda self:['python','/tmp/wrong.py'])
    with pytest.raises(ValueError,match='identity differs'):worker.assert_live_supervisor(manifest)
