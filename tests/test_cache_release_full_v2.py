import json
from pathlib import Path
import pytest
import torch
import numpy as np
from PIL import Image
from scripts.memory_token_io import digest

def test_full_seq11_audit_binds_observer_and_seq6_proof(tmp_path, monkeypatch):
    import json
    from scripts import audit_cache_release_full_v2 as full
    monkeypatch.setattr(full,'OUT',tmp_path)
    monkeypatch.setattr(full,'sources',lambda:{})
    monkeypatch.setattr(full,'warm_plan',lambda:{})
    monkeypatch.setattr(full,'validate_plan',lambda:{'fresh_reads':[{'sequence':'seq11'},{'sequence':'seq11'}]})
    monkeypatch.setattr(full,'verify_gate',lambda path:({'certificate_sha256':'cert'},'gate'))
    monkeypatch.setattr(full,'phase_artifact',lambda root,name,path:digest(path))
    phase_rows=[]; runs={}
    for mode,pid in [('reference',1),('capture',2),('historical',3),('oldcapture',4)]:
        folder=tmp_path/'seq11'/mode;folder.mkdir(parents=True)
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
            setup=tmp_path/'seq11'/(mode+'_setup_cache.jsonl')
            setup.write_text(''.join(json.dumps({'name':name,'mode':'clear_idle_cache','before':sample,'after':sample,
                                                 'driver_change_bytes':0})+'\n'
                            for name in ('after_model_load','after_state_init','after_prompt')))
            framefile=tmp_path/'seq11'/(mode+'_frame_cache.jsonl')
            framefile.write_text(''.join(json.dumps({'frame':x['frame'],'mode':'clear_idle_cache','before':sample,
                                      'after':sample,'driver_change_bytes':0})+'\n' for x in rows))
            run['cache_setup_sha256']=digest(setup);run['cache_frame_sha256']=digest(framefile)
        run['bounded_base_source_sha256']=digest(full.ROOT/'scripts/run_cache_release_warm_base_v2.py')
        if mode=='capture': run['reference_receipt_sha256']=digest(tmp_path/'seq11/reference/receipt.json')
        receipt=folder/'receipt.json';receipt.write_text(json.dumps(run));runs[mode]=run
        if mode in ('reference','capture'):
            side=folder/'bounded_receipt.json'
            side.write_text(json.dumps({'passed':True,'sequence':'seq11','mode':mode,
                'original_receipt_sha256':digest(receipt),'certificate_sha256':'cert','short_gate_sha256':'gate',
                'source_sha256':{},'label_guard':{'open_attempts':0},'peak_process_rss_bytes':123,
                'cache':{'final_sha256_verified':True,'evictions':1},'process_id':pid,'run_uuid':mode}))
            phase_rows.append({'phase':f'seq11_{mode}','bounded_receipt_sha256':digest(side),
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
    manifest={'sequence':'seq11','certificate_path':str(certfile),'certificate_sha256':'cert',
        'short_gate_path':str(gatefile),'short_gate_sha256':'gate','source_sha256':{},
        'memory_gate_path':'unused','memory_gate_sha256':'memory',
        'seq6_proof_path':'seq6-proof','seq6_proof_sha256':'proof',
        'phases':['seq11_reference','seq11_capture','seq11_audit'],'cache_frames':4,
        'min_disk_bytes':26*1024**3,'min_available_ram_bytes':4*1024**3,'max_process_rss_bytes':6*1024**3,
        'planned_capture_reads':2,'planned_payload_bytes':2*full.PER_READ,
        'source_snapshot_manifest_sha256':digest(snapshot_manifest)}
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest))
    oldref=tmp_path/'seq11/historical/receipt.json';oldcap=tmp_path/'seq11/oldcapture/receipt.json'
    certificate={'warm_gate_sha256':'warm','seq11_historical_reference':{'reference_path':str(oldref),
        'reference_sha256':digest(oldref)},'seq11_gate_capture':{'receipt_path':str(oldcap),'receipt_sha256':digest(oldcap)}}
    monkeypatch.setattr(full,'verify_certificate',lambda path:(certificate,'cert'))
    from scripts import run_cache_release_full_study_v2 as study
    monkeypatch.setattr(study,'verify_seq6_proof',lambda path,cert,gate:({},'proof'))
    from scripts import verify_mps_memory_diagnostic_v3 as memory_verifier
    monkeypatch.setattr(memory_verifier,'verify_gate',lambda path:({'certificate_sha256':'cert'},'memory'))
    monkeypatch.setattr(full,'checked_run',lambda root,seq,mode,*a:(runs[mode],digest(root/seq/mode/'receipt.json')))
    result=full.audit(tmp_path,'seq11',certfile,gatefile)
    assert result['passed'] and len(result['historical_observations'])==0
    assert result['minimum_observed_mask_dice']==1 and result['historical_traces'] is None
    monkeypatch.setattr(study,'verify_seq6_proof',lambda path,cert,gate:({},'wrong'))
    with pytest.raises(ValueError,match='Seq6 proof differs'):
        full.audit(tmp_path,'seq11',certfile,gatefile)
    monkeypatch.setattr(study,'verify_seq6_proof',lambda path,cert,gate:({},'proof'))
    with pytest.raises(ValueError,match='manifest differs'):
        full.audit(tmp_path,'seq8',certfile,gatefile)
    checks=iter([({'certificate_sha256':'cert'},'memory'),({'certificate_sha256':'cert'},'altered')])
    monkeypatch.setattr(memory_verifier,'verify_gate',lambda path:next(checks))
    with pytest.raises(ValueError,match='Cache-release proof changed during audit'):
        full.audit(tmp_path,'seq11',certfile,gatefile)
    monkeypatch.setattr(memory_verifier,'verify_gate',lambda path:({'certificate_sha256':'cert'},'memory'))
    monkeypatch.setattr(memory_verifier,'verify_gate',lambda path:({'certificate_sha256':'cert'},'wrong'))
    with pytest.raises(ValueError,match='MPS memory diagnostic proof differs'):
        full.audit(tmp_path,'seq11',certfile,gatefile)
    monkeypatch.setattr(memory_verifier,'verify_gate',lambda path:({'certificate_sha256':'cert'},'memory'))
    trace=tmp_path/'seq11/capture_frame_cache.jsonl';original=trace.read_text()
    records=[json.loads(x) for x in original.splitlines()];records[0]['driver_change_bytes']=1
    trace.write_text(''.join(json.dumps(x)+'\n' for x in records))
    runs['capture']['cache_frame_sha256']=digest(trace)
    with pytest.raises(ValueError,match='Cache-release delta differs'):
        full.audit(tmp_path,'seq11',certfile,gatefile)
    trace.write_text(original);runs['capture']['cache_frame_sha256']=digest(trace)
    snapshot_file=snapshot/'external/Lean-SAM2-token-window/sam2/utils/misc.py'
    snapshot_file.write_bytes(b'tamper')
    with pytest.raises(ValueError,match='source snapshot changed'):
        full.audit(tmp_path,'seq11',certfile,gatefile)



def test_worker_cli_accepts_only_the_seven_remaining_sequences(tmp_path,monkeypatch):
    import sys
    from scripts import run_cache_release_full_v2 as worker
    monkeypatch.setattr(worker,'install_mask_open_guard',lambda _: {'open_attempts':0})
    for sequence in ('seq8','seq11','seq16','seq17','seq18','seq19','seq21'):
        monkeypatch.setattr(sys,'argv',['worker','--sequence',sequence,'--mode','reference',
            '--output',str(tmp_path/'out'),'--certificate',str(tmp_path/'cert'),
            '--short-gate',str(tmp_path/'gate'),'--batch-root',str(tmp_path/'missing')])
        with pytest.raises(FileNotFoundError):worker.main()
    monkeypatch.setattr(sys,'argv',['worker','--sequence','seq6','--mode','reference',
        '--output',str(tmp_path/'out'),'--certificate',str(tmp_path/'cert'),
        '--short-gate',str(tmp_path/'gate'),'--batch-root',str(tmp_path/'missing')])
    with pytest.raises(SystemExit):worker.main()


def test_v2_source_closure_and_disabled_execution():
    from scripts import run_cache_release_full_v2 as worker
    from scripts import run_cache_release_full_study_v2 as study
    assert worker.ENABLE_REVIEWED_EXECUTION and study.ENABLE_REVIEWED_EXECUTION
    sources=worker.sources()
    for name in ('scripts/run_cache_release_full_v2.py','scripts/run_cache_release_full_study_v2.py',
                 'scripts/audit_cache_release_full_v2.py','scripts/run_cache_release_warm_base_v2.py',
                 'scripts/run_cache_release_full_study_v1.py','scripts/verify_mps_memory_diagnostic_v3.py',
                 'protocols/CACHE_RELEASE_REMAINING_V2.md'):
        assert sources[name] == digest(worker.ROOT/name)


def test_direct_entrypoint_disabled(tmp_path, monkeypatch):
    from scripts import run_cache_release_full_v2 as worker
    monkeypatch.setattr(worker,'ENABLE_REVIEWED_EXECUTION',False)
    with pytest.raises(ValueError,match='disabled pending review'):
        worker.run_capture(sequence='seq11',mode='reference',output=tmp_path/'out',
                           certificate_path='missing',short_gate_path='missing')


def test_real_seven_sequence_geometry_and_plan_without_label_opens():
    import subprocess, sys
    program = """from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.crossfit_observer_capture_shared_v1 import DATA, validate_plan
from scripts.run_cache_release_warm_base_v2 import warm_plan, video_inputs
guard=install_mask_open_guard(DATA); index=validate_plan(); warm=warm_plan()
expected={'seq8':[0,0,1920,1080], 'seq11':[0,0,720,576], 'seq16':[0,0,1280,720],
          'seq17':[0,0,1280,720], 'seq18':[0,0,1280,720], 'seq19':[0,0,1280,720], 'seq21':[0,0,1280,720]}
for seq,expected_roi in expected.items():
    frames,records,start,targets,mask,box,points,roi=video_inputs(seq,index,warm)
    assert targets==[x['frame'] for x in index['fresh_reads'] if x['sequence']==seq]
    assert len(targets)==12 and roi==expected_roi and len(frames)==len(records)
    assert mask.shape==(expected_roi[3]-expected_roi[1],expected_roi[2]-expected_roi[0])
assert guard['open_attempts']==0
"""
    result=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True)
    assert result.returncode==0, result.stderr
