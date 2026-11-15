import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import run_remaining_boundary_factorial_v1 as launcher
from scripts import score_remaining_boundary_factorial_v1 as scorer
from scripts import audit_remaining_boundary_factorial_v1 as auditor
from scripts import audit_remaining_boundary_video_short_v1 as short_auditor


def test_commands_are_8_complete_then_score(tmp_path):
    phases, mapping = launcher.commands(tmp_path, Path('/pre'), Path('/old-gate'),
                                        Path('/new-gate'), Path('/audit'))
    assert len(phases) == 9 and phases[-1][0] == 'score'
    assert [name for name,_ in phases[:-1]] == [f'{s}_{a}' for s in launcher.SEQUENCES for a in launcher.ARMS]
    assert set(mapping) == set(launcher.SEQUENCES)
    for seq in launcher.SEQUENCES:
        for arm in launcher.ARMS:
            cmd = dict(phases)[f'{seq}_{arm}']
            assert ('--matched-a' in cmd) is (arm != 'A')
            assert cmd[cmd.index('--extended-preflight')+1] == '/pre'
            assert cmd[cmd.index('--extension-gate')+1] == '/new-gate'
            if arm != 'A': assert cmd[cmd.index('--matched-a')+1] == mapping[seq]['A']


def test_phase_environment_overrides_unsafe_settings(monkeypatch):
    monkeypatch.setenv('PYTORCH_MPS_FAST_MATH','1')
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    monkeypatch.setenv('PYTHONHASHSEED','random')
    env = launcher.phase_environment()
    assert [env[k] for k in ('PYTORCH_MPS_FAST_MATH','PYTORCH_ENABLE_MPS_FALLBACK','PYTHONHASHSEED')] == ['0','0','0']


def test_launcher_rejects_inputs_outside_experiment_root(tmp_path):
    root=tmp_path/'root'; root.mkdir(); inside=root/'gate'; inside.write_text('{}')
    outside=tmp_path/'outside'; outside.write_text('{}')
    launcher.validate_input_roots((inside,),root)
    with pytest.raises(ValueError,match='outside'): launcher.validate_input_roots((inside,outside),root)


def test_map_rejects_missing_alias_and_escaping_paths(tmp_path):
    root = tmp_path/'root'; root.mkdir(); run = root/'run'; run.mkdir()
    data = {}
    for seq in scorer.SEQUENCES:
        data[seq] = {}
        for arm in scorer.ARMS:
            folder = run/seq/arm; folder.mkdir(parents=True)
            data[seq][arm] = str(folder)
    path = run/'run_map.json'; path.write_text(json.dumps(data))
    assert len(scorer.mapped_runs(path,root)) == 2
    data['seq6']['B'] = data['seq6']['A']; path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='distinct'): scorer.mapped_runs(path,root)
    del data['seq6']; path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='both'): scorer.mapped_runs(path,root)


def test_map_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path/'map.json'; path.write_text('{"seq6":{},"seq6":{}}')
    with pytest.raises(ValueError,match='Duplicate'): scorer.mapped_runs(path,tmp_path)


def test_full_validation_precedes_any_label_access(tmp_path,monkeypatch):
    root=tmp_path/'root'; root.mkdir(); path=root/'map.json'; path.write_text('{}')
    monkeypatch.setattr(scorer,'BASE',root)
    monkeypatch.setattr(scorer,'mapped_runs',lambda *_: {})
    monkeypatch.setattr(scorer,'load_manifest',lambda *_: {})
    monkeypatch.setattr(scorer,'verify_launch',lambda *_: (_ for _ in ()).throw(ValueError('unfinished')))
    monkeypatch.setattr(scorer,'load_label_mask',lambda *_: (_ for _ in ()).throw(AssertionError('label read')))
    monkeypatch.setattr(scorer.sys,'argv',['score','--experiment-root',str(root),'--run-map',str(path),'--output',str(root/'out')])
    with pytest.raises(ValueError,match='unfinished'): scorer.main()


def test_metric_helper_tamper_rejected_before_any_label_read(tmp_path, monkeypatch):
    from scripts.memory_token_io import digest
    checkpoint=tmp_path/'checkpoint'; checkpoint.write_bytes(b'checkpoint')
    detector=tmp_path/'detector'; detector.write_bytes(b'detector')
    metric=tmp_path/'metric.py'; metric.write_text('original')
    mapping={s:{a:str(tmp_path/s/a) for a in scorer.ARMS} for s in scorer.SEQUENCES}
    run_map=tmp_path/'run_map.json'; run_map.write_text(json.dumps(mapping))
    manifest={'run_map':mapping,'sequences':list(scorer.SEQUENCES),'arms':list(scorer.ARMS),
              'python_hash_seed':'0','mps_fallback':'0','mps_fast_math':'0',
              'checkpoint_sha256':digest(checkpoint),'detector_sha256':digest(detector),
              'source_sha256':{'primary_metric_helper':digest(metric)}}
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(scorer,'CHECKPOINT',checkpoint)
    monkeypatch.setattr(scorer,'DETECTOR',detector)
    monkeypatch.setattr(scorer,'DETECTOR_SHA256',digest(detector))
    monkeypatch.setattr(scorer,'study_sources',lambda:{'primary_metric_helper':metric})
    monkeypatch.setattr(scorer,'load_label_mask',lambda *_: (_ for _ in ()).throw(AssertionError('label read')))
    metric.write_text('tampered')
    with pytest.raises(ValueError,match='Launch sources changed'):
        scorer.verify_launch(run_map,mapping)


def test_preflight_sequence_selection_checked_before_labels(tmp_path):
    pre=tmp_path/'pre'; pre.mkdir(); (pre/'manifest.json').write_text(json.dumps({'selection':['seq6']}))
    with pytest.raises(ValueError,match='selection'): scorer.verify_runs({}, {}, pre, {})


def test_native_metric_helpers_return_expected_values():
    gt=np.array([[1,0],[1,0]],dtype=bool); pred=np.array([[1,1],[0,0]],dtype=bool)
    result=scorer.metric(pred,gt)
    assert result['dice'] == .5 and result['iou'] == pytest.approx(1/3)
    assert result['precision'] == .5 and result['recall'] == .5


def test_independent_audit_does_not_open_labels_before_precheck(monkeypatch):
    monkeypatch.setattr(auditor,'precheck',lambda *_: (_ for _ in ()).throw(ValueError('incomplete')))
    monkeypatch.setattr(auditor,'label',lambda *_: (_ for _ in ()).throw(AssertionError('label read')))
    with pytest.raises(ValueError,match='incomplete'): auditor.audit('/unused')


def test_independent_boundary_metric_is_not_scorer_implementation():
    a=np.zeros((15,15),dtype=bool); b=np.zeros_like(a)
    a[3:10,3:10]=1; b[4:11,3:10]=1
    actual=auditor.metrics(a,b); expected=scorer.metric(a,b)
    for key in ('dice','iou','precision','recall','boundary_f_5px'):
        assert actual[key] == pytest.approx(expected[key])


def test_independent_telemetry_checks_coverage_and_presence_values():
    t={'min':-1024.,'max':-100.,'numel':4,'positive_pixels':0,'sha256':'a'*64}
    row={'conditioning':False,'frame':1,'object_score_logit':-2.,'native_object_present':False,
         'native_exact_suppressed':True,**{k:dict(t) for k in ('native','refined','output','memory_high_res')}}
    assert auditor.telemetry_stats([row],0,2)['native_absence_frames']==1
    with pytest.raises(ValueError,match='coverage'): auditor.telemetry_stats([row],0,3)
    row['object_score_logit']=float('nan')
    with pytest.raises(ValueError,match='presence'): auditor.telemetry_stats([row],0,2)


def test_short_audit_rejects_failed_gate_before_artifact_reads(tmp_path):
    gate=tmp_path/'gate.json'; gate.write_text(json.dumps({'passed':False}))
    with pytest.raises(ValueError,match='Short gate failed'):
        short_auditor.audit(gate,tmp_path,gate,gate,gate)


def test_short_audit_checks_frozen_preflight_and_rejects_prompt_tamper(tmp_path,monkeypatch):
    from scripts import preflight_remaining_boundary_video_v1 as preflight_module
    from scripts import resume_source_group_capture_v1 as resume
    from scripts.memory_token_io import digest
    monkeypatch.setattr(resume,'BASE',tmp_path)
    folder=tmp_path/'pre'; folder.mkdir(); (folder/'manifest.json').write_text('{}')
    old_gate=tmp_path/'old.json'
    checkpoint=tmp_path/'checkpoint'; checkpoint.write_bytes(b'weights')
    detector=tmp_path/'detector'; detector.write_bytes(b'detector')
    fingerprint=tmp_path/'fingerprint.json'; fingerprint.write_text('{}')
    runtime={'passed':True,'candidate_checkpoint_sha256':'candidate',
             **{name+'_tensor_fingerprints_sha256':digest(fingerprint) for name in ('model','refiner')}}
    old_gate.write_text(json.dumps({'passed':True,'runtime_identity':runtime}))
    prompt_meta=tmp_path/'prompt.json'; prompt_meta.write_text('{}')
    prompt_mask=tmp_path/'prompt.png'; Image.fromarray(np.ones((4,4),dtype=np.uint8)*255).save(prompt_mask)
    input_image=tmp_path/'input.png'; Image.fromarray(np.zeros((4,4),dtype=np.uint8)).save(input_image)
    pre={'sequences':{seq:{'first_detection_index':0,'prompt_metadata_path':str(prompt_meta),
          'prompt_mask_path':str(prompt_mask),'prompt_metadata_sha256':digest(prompt_meta),'prompt_mask_sha256':digest(prompt_mask),
          'view':{'kind':'full_source_frame'},'image_size_wh':[4,4],
          'frames':[{'frame':i,'source_frame_index':i,'image_path':str(input_image),'image_sha256':digest(input_image)} for i in range(5)]}
          for seq in short_auditor.SEQUENCES}}
    monkeypatch.setattr(preflight_module,'load_extended_preflight',lambda *_:pre)
    sources={'runner_sha256':'scripts/run_remaining_boundary_refiner_v1.py',
             'routing_sha256':'modeling/lean_boundary_refinement_routing_v1.py',
             'runtime_sha256':'modeling/lean_boundary_window_runtime_v1.py',
             'verifier_sha256':'scripts/verify_remaining_boundary_video_short_gate_v1.py',
             'preflight_source_sha256':'scripts/preflight_remaining_boundary_video_v1.py'}
    gate={'passed':True,**{k:digest(short_auditor.ROOT/v) for k,v in sources.items()},
          'extended_preflight_sha256':digest(folder/'manifest.json'),
          'engineering_gate_sha256':digest(old_gate),'runs':{}}
    for seq in short_auditor.SEQUENCES:
        gate['runs'][seq]={}
        for arm in short_auditor.ARMS:
            out=tmp_path/seq/arm; (out/'verification').mkdir(parents=True)
            for name in ('model','refiner'): (out/'verification'/f'{name}_tensor_fingerprints.json').write_text('{}')
            actual_runtime={'passed':True,'candidate_checkpoint_sha256':'candidate',
                    **{name+'_tensor_fingerprints_sha256':digest(out/'verification'/f'{name}_tensor_fingerprints.json')
                       for name in ('model','refiner')}}
            runtime=actual_runtime
            (out/'verification/runtime_verification.json').write_text(json.dumps(actual_runtime))
            for name in ('memory_events.json','positions.json','selections.json'): (out/name).write_text('[]')
            logit={'positive_pixels':0}; telemetry=[{'conditioning':False,'frame':i,
                'native_object_present':True,'refined':logit} for i in range(1,5)]
            (out/'telemetry.json').write_text(json.dumps(telemetry))
            rows=[]; state=[]
            for i in range(5):
                file=out/f'{i:06d}.png'; Image.fromarray(np.zeros((4,4),dtype=np.uint8)).save(file)
                rows.append({'frame':i,'sha256':digest(file)})
                output='refined' if i==1 and arm in ('B','D') else f'original{i}'
                memory='corrected' if i==1 and arm in ('C','D') else f'mem{i}'
                state.append({'frame':i,'mask_sha256':digest(file),'pred_masks_sha256':f'pred{i}',
                    'maskmem_features_sha256':memory,'obj_ptr_sha256':f'ptr{i}',
                    'object_score_logits_sha256':f'score{i}','output_logits_sha256':output})
            rec={'passed':True,'sequence':seq,'arm':'A' if arm=='A_repeat' else arm,'limit':4,'full_video':False,
                 'label_guard':{'open_attempts':0},'extended_preflight_sha256':digest(folder/'manifest.json'),
                 'engineering_gate_sha256':digest(old_gate),'extension_gate_sha256':None,'extension_audit_sha256':None,
                 'prompt_frame':0,'prompt_sha256':digest(prompt_meta),'prompt_mask_sha256':digest(prompt_mask),'view':pre['sequences'][seq]['view'],
                 'input_images':[{'frame':i,'source_frame_index':i,'sha256':digest(input_image)} for i in range(5)],
                 'checkpoint_sha256':digest(checkpoint),'detector_sha256':digest(detector),
                 'candidate_checkpoint_sha256':'candidate','runner_source_sha256':gate['runner_sha256'],
                 'masks':rows,'post_prompt_state':state}
            for k,n in short_auditor.FILES: rec[k]=digest(out/n)
            (out/'completion.json').write_text(json.dumps(rec))
            gate['runs'][seq][arm]={'folder':str(out),'completion_sha256':digest(out/'completion.json'),
                'runtime_sha256':digest(out/'verification/runtime_verification.json')}
    gate['runtime_identity']=runtime
    gate_file=tmp_path/'gate.json'; gate_file.write_text(json.dumps(gate))
    assert short_auditor.audit(gate_file,folder,old_gate,checkpoint,detector)['runs_verified']==10
    victim=Path(gate['runs']['seq6']['C']['folder'])/'completion.json'
    rec=json.loads(victim.read_text()); rec['prompt_sha256']='forged'; victim.write_text(json.dumps(rec))
    gate['runs']['seq6']['C']['completion_sha256']=digest(victim); gate_file.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='scope or label guard'):
        short_auditor.audit(gate_file,folder,old_gate,checkpoint,detector)

