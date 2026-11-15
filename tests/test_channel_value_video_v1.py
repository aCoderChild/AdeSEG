"""CPU-only tests; real MPS behavior remains the smoke gate's job."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch

from modeling import lean_channel_value_window_runtime_v1 as runtime
from modeling.lean_channel_value_routing_v1 import route_channel_value_refinement
from modeling.seq23_viewport import VIEWPORT,restore_mask
from scripts import run_channel_value_video_study_v1 as launcher
from scripts.run_channel_value_video_v1 import compare_archived
from scripts import score_channel_value_video_v1 as scorer
from scripts.memory_token_io import digest
from test_boundary_video_routing_v1 import NumericPredictor
from test_lean_contrast_refinement_integration import Refiner,run


@pytest.mark.parametrize('arm,out_delta,mem_delta',[('A',0,0),('B',1,0),('C',0,1),('D',1,1)])
def test_routing_preserves_conditioning_and_native_score(arm,out_delta,mem_delta):
    model=NumericPredictor().eval();original,old=run(model);telemetry=[]
    with route_channel_value_refinement(model,Refiner().eval(),arm,test_only=True,telemetry=telemetry):
        compact,out=run(model)
        assert torch.equal(compact['pred_masks'],original['pred_masks'])
        assert torch.equal(out,old+out_delta)
        assert torch.allclose(compact['memory'],original['memory']+mem_delta)
        conditioned,cout=run(model,conditioning=True)
        assert torch.equal(cout,old) and torch.equal(conditioned['memory'],original['memory'])
    assert telemetry[0]['object_score_logit']==-2 and not telemetry[0]['native_object_present']
    assert '_track_step' not in vars(model)


def test_nine_run_protocol_freezes_full_videos_before_scoring(tmp_path):
    names=[name for name,_ in launcher.phases(tmp_path)]
    assert names[:7]==['certificate','smoke_seq20_A','smoke_seq20_B','smoke_seq20_C','smoke_seq20_D',
                       'smoke_seq23_A','smoke_audit']
    assert len([x for x in names if x.startswith('full_')])==12
    assert names[-3:]==['verify','score','audit']
    assert [x for x in names if x.startswith('full_seq10_')]==['full_seq10_B','full_seq10_C','full_seq10_D']


def test_seq23_viewport_is_restored_without_the_inset():
    mask=np.ones((VIEWPORT[3]-VIEWPORT[1],VIEWPORT[2]-VIEWPORT[0]),dtype=bool)
    full=restore_mask(mask)
    assert full.shape==(720,1280) and not full[:,:VIEWPORT[0]].any()
    assert full[VIEWPORT[1]:VIEWPORT[3],VIEWPORT[0]:VIEWPORT[2]].all()


def test_recorded_b_state_is_checked_and_c_memory_may_differ(tmp_path):
    original={'frame':0,'mask_sha256':'init','output_logits_sha256':'output0','pred_masks_sha256':'p0',
              'maskmem_features_sha256':'m0','obj_ptr_sha256':'u0','object_score_logits_sha256':'s0'}
    later={'frame':1,'mask_sha256':'mask1','output_logits_sha256':'out1','pred_masks_sha256':'p1',
           'maskmem_features_sha256':'m1','obj_ptr_sha256':'u1','object_score_logits_sha256':'s1'}
    old={'post_prompt_state':[original,later]};trace={}
    baseline=tmp_path/'completion.json';baseline.write_text('{}')
    for key in ('memory_events','positions','selections'):
        file=tmp_path/(key+'.json');file.write_text('[]');old[key+'_sha256']=digest(file);trace[key]=[]
    changed=copy.deepcopy([original,later]);changed[1]['output_logits_sha256']='refined';changed[1]['mask_sha256']='refined'
    compare_archived(changed,trace,baseline,old,'B',False)
    changed[1]['maskmem_features_sha256']='unexpected'
    with pytest.raises(ValueError,match='native state'):compare_archived(changed,trace,baseline,old,'B',False)
    c=copy.deepcopy([original,later]);c[1]['maskmem_features_sha256']='changed memory'
    compare_archived(c,trace,baseline,old,'C',False)


def test_active_phase_stops_when_reserve_is_crossed(tmp_path,monkeypatch):
    events=[]
    class FakeProcess:
        def __init__(self,*args,**kwargs):pass
        def wait(self,timeout=None):
            if not events:raise subprocess.TimeoutExpired('child',timeout)
            return -15
        def terminate(self):events.append('terminated')
        def kill(self):events.append('killed')
    monkeypatch.setattr(launcher.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(launcher.shutil,'disk_usage',lambda _:SimpleNamespace(free=launcher.GUARD-1))
    with pytest.raises(RuntimeError,match='reserve crossed'):
        launcher.run_phase(['child'],tmp_path/'phase.log',tmp_path,[])
    assert events==['terminated']


def test_no_object_telemetry_is_counted_without_gating():
    def item(frame,conditioning,present,refined_positive):
        part=lambda n:{'min':-1.,'max':1.,'positive_pixels':n,'numel':4,'sha256':'a'*64}
        return {'frame':frame,'conditioning':conditioning,'native_object_present':present,
                'native_exact_suppressed':not present,'object_score_logit':1. if present else -1.,
                'native':part(0),'refined':part(refined_positive),'output':part(refined_positive),
                'memory_high_res':part(0)}
    counts=scorer.telemetry_counts([item(3,True,True,0),item(4,False,False,1),item(5,False,True,0)],3,6)
    assert counts['native_absent_score_frames']==1
    assert counts['refined_positive_with_native_absent_score']==1
    assert counts['output_positive_with_native_absent_score']==1


def test_resume_reuses_phase_bound_prefix_and_preserves_partial(tmp_path,monkeypatch):
    from scripts import verify_channel_value_video_smoke_v1 as smoke
    monkeypatch.setattr(launcher,'OUT',tmp_path)
    monkeypatch.setattr(launcher,'runtime_sources',lambda:{'runtime':'frozen'})
    monkeypatch.setattr(runtime,'verify_certificate',lambda *args:None)
    monkeypatch.setattr(smoke,'verify_smoke_audit',lambda *args,**kwargs:None)
    names=[n for n,_ in launcher.phases(tmp_path)]
    manifest={'sources':{'source':'frozen'},'runtime_sources':{'runtime':'frozen'},'phases':names,
              'label_guard':{'open_attempts':0}}
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest))
    rows=[]
    for name in names[:8]:
        file=launcher.artifact(tmp_path,name);file.parent.mkdir(parents=True,exist_ok=True);file.write_text('{}')
        log=tmp_path/(name+'.log');log.write_text('done')
        rows.append({'phase':name,'passed':True,'artifact_path':str(file),
                     'artifact_sha256':digest(file),'log_sha256':digest(log)})
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    _,_,count=launcher.validate_resume(tmp_path,{'source':'frozen'});assert count==8
    next_name=names[8];partial=launcher.artifact(tmp_path,next_name).parent;partial.mkdir(parents=True)
    (partial/'data').write_text('keep');(tmp_path/(next_name+'.log')).write_text('failed')
    archived=launcher.preserve_partial(tmp_path,next_name)
    assert (archived/'output/data').read_text()=='keep' and (archived/'phase.log').read_text()=='failed'
    assert not partial.exists()
    launcher.artifact(tmp_path,names[7]).write_text('tampered')
    with pytest.raises(ValueError,match='phase artifact changed'):
        launcher.validate_resume(tmp_path,{'source':'frozen'})


def test_certificate_fast_revalidation_fails_closed_on_tampering(tmp_path,monkeypatch):
    root=tmp_path/'output';study=root/'study';manifest_dir=root/'source_group_boundary_factorial_v1_bvqxd475/manifest'
    ck=study/'runs/foreground_background_asinh/checkpoint.pt';plan=study/'plan/plan.json'
    legacy=root/'legacy/seq20/A/verification/model_tensor_fingerprints.json'
    for f in (ck,plan,legacy,manifest_dir/'manifest.json'):f.parent.mkdir(parents=True,exist_ok=True)
    manifest={'feature_pipeline_sha256':'feature'};(manifest_dir/'manifest.json').write_text(json.dumps(manifest))
    fixed={'manifest':digest(manifest_dir/'manifest.json'),'parity':'p'}
    pl={'dependency_sha256':{'dep':'ok'},'manifest_sha256':fixed['manifest'],'environment':{'device':'mps'}}
    plan.write_text(json.dumps(pl));fixed['plan']=digest(plan)
    payload={'study_arm':'foreground_background_asinh','plan_sha256':fixed['plan'],'parity_sha256':'p',
             'dependency_sha256':{'dep':'ok'}}
    torch.save(payload,ck);fixed['checkpoint']=digest(ck)
    legacy.write_text('{}');legacy_sha=digest(legacy)
    monkeypatch.setattr(runtime,'OUT',root);monkeypatch.setattr(runtime,'STUDY',study)
    monkeypatch.setattr(runtime,'LEGACY',root/'legacy');monkeypatch.setattr(runtime,'LEGACY_MODEL_SHA',legacy_sha)
    monkeypatch.setattr(runtime,'HASHES',fixed);monkeypatch.setattr(runtime,'deps',lambda:{'dep':'ok'})
    monkeypatch.setattr(runtime,'runtime_sources',lambda:{'source':'ok'})
    proof={'passed':True,'study':str(study),'study_hashes':fixed,'runtime_sources':{'source':'ok'},
           'training_dependencies':{'dep':'ok'},'label_guard':{'open_attempts':0},
           'legacy_model_fingerprint_sha256':legacy_sha,'verifier_sha256':digest(runtime.ROOT/'scripts/verify_channel_value_video_study_v1.py'),
           'training_environment':pl['environment'],'feature_pipeline_sha256':'feature'}
    certificate=root/'certificate.json';certificate.write_text(json.dumps(proof));sha=digest(certificate)
    # Per-child verification must not reread the inactive historical tensors.
    monkeypatch.setattr(runtime,'load_prepared',lambda *_:pytest.fail('unexpected full capture rehash'))
    assert runtime.verify_certificate(certificate,sha)[2]['study_arm']=='foreground_background_asinh'
    monkeypatch.setattr(runtime,'runtime_sources',lambda:{'source':'changed'})
    with pytest.raises(ValueError,match='certificate or source'):
        runtime.verify_certificate(certificate,sha)
    monkeypatch.setattr(runtime,'runtime_sources',lambda:{'source':'ok'})
    ck.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checkpoint changed'):
        runtime.verify_certificate(certificate,sha)


def test_synthetic_full_canvas_scoring_reports_macro_and_frame_means(tmp_path,monkeypatch):
    plan={};all_inputs={};proof={'inputs':{},'receipts':{},'references':{}}
    for seq in scorer.SEQUENCES:
        count=3 if seq=='seq10' else 2;shape=(9,15) if seq=='seq23' else (9,9)
        frames=[];metadata=[];records=[];baseline=tmp_path/'archived'/seq/'A';baseline.mkdir(parents=True)
        masks={a:[] for a in scorer.ARMS}
        for i in range(count):
            image=tmp_path/f'{seq}_{i}_image.png';Image.fromarray(np.zeros((*shape,3),np.uint8)).save(image)
            truth=np.zeros(shape,np.uint8);truth[2:5,-4:-1]=1
            target=tmp_path/f'{seq}_{i}_target.png';Image.fromarray(truth).save(target)
            frames.append(SimpleNamespace(image_path=image,mask_path=target,frame_index=i))
            metadata.append({'label_sha256':digest(target)});records.append({'image_sha256':digest(image)})
            for arm in scorer.ARMS:
                folder=baseline if arm=='A' else tmp_path/'full'/seq/arm;folder.mkdir(parents=True,exist_ok=True)
                pred=truth.copy() if i>0 and not(seq=='seq23' and arm=='A') else np.zeros(shape,np.uint8)
                file=folder/f'{i:06d}.png';Image.fromarray(pred*255).save(file)
                masks[arm].append({'frame':i,'sha256':digest(file)})
        meta=tmp_path/f'{seq}_frames.json';meta.write_text(json.dumps(metadata))
        baseline_file=baseline/'completion.json';baseline_file.write_text(json.dumps({'masks':masks['A']}))
        proof['references'][seq]={'sha256':digest(baseline_file)};proof['receipts'][seq]={}
        for arm in ('B','C','D'):
            file=tmp_path/'full'/seq/arm/'receipt.json';file.write_text(json.dumps({'masks':masks[arm]}))
            proof['receipts'][seq][arm]={'sha256':digest(file)}
        proof['inputs'][seq]={'label_metadata':{'path':str(meta),'sha256':digest(meta)},'images':records}
        all_inputs[seq]=(frames,None,0,None,None,baseline_file,None,None)
    monkeypatch.setattr(scorer,'validate_plan',lambda:plan)
    monkeypatch.setattr(scorer,'sequence_inputs',lambda _plan,seq:all_inputs[seq])
    primary=scorer.compute(tmp_path,proof);independent=scorer.compute(tmp_path,proof,independent=True)
    errors=[];scorer.compare(independent,primary,[],errors);assert not errors
    assert primary['overall']['A']['patients_with_present_frames']==4
    assert primary['overall']['A']['patient_macro_present_dice']==pytest.approx(.75)
    assert primary['overall']['A']['present_frames']==5
    assert primary['overall']['A']['frame_weighted_present_dice']==pytest.approx(.8)
    assert primary['summaries']['seq23']['A']['all']['present_frames']==2
    assert primary['summaries']['seq23']['A']['post_prompt']['present_frames']==1


def test_verification_failure_does_not_open_target_masks(tmp_path,monkeypatch):
    tmp_path.mkdir(exist_ok=True);attempts=[]
    monkeypatch.setattr(scorer,'install_mask_open_guard',lambda _:{'open_attempts':0})
    monkeypatch.setattr(scorer,'validate_frozen',lambda *_:(_ for _ in ()).throw(ValueError('not frozen')))
    monkeypatch.setattr(scorer,'load_label_mask',lambda *_:attempts.append('label'))
    monkeypatch.setattr(scorer.sys,'argv',['score','--mode','verify','--batch-root',str(tmp_path),'--output',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='not frozen'):scorer.main()
    assert not attempts and not (tmp_path/'out').exists()
