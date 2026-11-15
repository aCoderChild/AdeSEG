import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts import channel_value_fixed_read_v1 as study
from scripts import run_channel_value_fixed_read_v1 as launcher
from scripts.memory_token_io import digest


def test_plan_has_matched_initial_states_and_discloses_smaller_reference(monkeypatch):
    monkeypatch.setattr(study, 'deps', lambda: {'source': 'fixed'})
    monkeypatch.setattr(study, 'schedule', lambda samples: [['same order']])
    monkeypatch.setattr(study, 'require_environment_flags', lambda: {'seed': '0'})
    manifest={'feature_pipeline_sha256':'f','feature_pipeline':{'runtime_environment':{'device':'mps'}},'samples':[]}
    plan=study.build_plan(manifest,'m')
    assert len(set(plan['initial_state_sha256'][a] for a in study.ARMS[:-1]))==1
    assert all(plan['parameter_count'][a]==63809 for a in study.ARMS[:-1])
    assert plan['parameter_count']['region_reference']==48005
    assert 'not parameter-matched' in plan['limitations'][0]


def test_launcher_freezes_all_arms_before_replay_and_scoring(tmp_path):
    names=[name for name,_ in launcher.phases(tmp_path)]
    assert names==['plan','parity']+['train_'+a for a in study.ARMS]+['replay','score','audit']


def test_real_214_manifest_sha_is_not_a_shadowed_receipt_sha():
    if not (launcher.MANIFEST/'manifest.json').exists():
        pytest.skip('Original audited 214-read manifest unavailable')
    code='''
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v2 import load_prepared
from scripts.run_channel_value_fixed_read_v1 import MANIFEST,ROOT
from scripts import run_channel_value_fixed_read_v1 as launcher, channel_value_fixed_read_v1 as study
from scripts.memory_token_io import digest
assert study.load_prepared is launcher.load_prepared is load_prepared
g=install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl')
m,sha=study.load_prepared(MANIFEST)
assert len(m['samples'])==214 and sha==digest(MANIFEST/'manifest.json')
assert g['open_attempts']==0
'''
    subprocess.run([sys.executable,'-c',code],check=True)


def test_real_training_sentinel_is_suppressed_and_positive():
    if not (launcher.MANIFEST/'manifest.json').exists():
        pytest.skip('Original audited 214-read manifest unavailable')
    code='''
import torch
from scripts.source_group_training_label_guard_v1 import install_training_mask_guard
from scripts.run_channel_value_fixed_read_v1 import MANIFEST,ROOT
from scripts.channel_value_fixed_read_v1 import load_prepared,inputs,checked_target
g=install_training_mask_guard(ROOT/'data/polypgen_sequence.jsonl')
m,_=load_prepared(MANIFEST)
x=next(x for x in m['samples'] if x['split']=='train' and x['sequence']=='seq4' and x['frame']==24)
args,shape=inputs(x,'cpu');target=checked_target(x,shape)
assert bool(torch.all(args[2]==-1024)) and bool(torch.any(target>0))
assert g['blocked_open_attempts']==0
'''
    subprocess.run([sys.executable,'-c',code],check=True)


def test_checkpoint_modes_and_config_are_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(study,'deps',lambda:{'source':'fixed'})
    plan=tmp_path/'plan';plan.write_text('p')
    parity=tmp_path/'parity';parity.write_text('q')
    sha='a'*64;manifest={'checkpoint_sha256':sha,'feature_pipeline_sha256':sha}
    for arm in study.ARMS:
        net=study.model_for(arm)
        payload=study.payload_for(net,arm,manifest,sha,plan,parity,1)
        loaded=study.load_model(payload,arm,manifest,sha)
        assert sum(x.numel() for x in loaded.parameters()) == (48005 if arm=='region_reference' else 63809)
        if arm!='region_reference':
            wrong=copy.deepcopy(payload);wrong['config']['attention_size']=16
            with pytest.raises(ValueError,match='geometry'):
                study.load_model(wrong,arm,manifest,sha)


def test_score_rejects_unfrozen_replay_before_opening_labels(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'outpath',lambda path:tmp_path/'unused')
    monkeypatch.setattr(study,'require_plan',lambda *args:{})
    monkeypatch.setattr(study,'require_parity',lambda *args:None)
    monkeypatch.setattr(study,'phase_inputs',lambda *args:['p','q'])
    monkeypatch.setattr(study,'digest',lambda *args:'p' if str(args[0]).endswith('plan') else 'q')
    calls=[];monkeypatch.setattr(study,'ensure_target',lambda *args:calls.append('label'))
    (tmp_path/'phases.jsonl').write_text(json.dumps({'phase':'plan','passed':True})+'\n')
    guard={'active':True,'open_attempts':0}
    with pytest.raises(ValueError,match='Replay phase'):
        study.score(tmp_path,{},'sha',tmp_path/'plan',tmp_path/'parity',tmp_path/'v',tmp_path/'out',guard)
    assert calls==[] and guard['active']


def test_target_rehashes_after_label_decode(tmp_path,monkeypatch):
    label=tmp_path/'label.png';label.write_bytes(b'original')
    item={'label':str(label),'label_sha256':digest(label)}
    def decode(*args):
        label.write_bytes(b'changed')
        return torch.tensor(1)
    monkeypatch.setattr(study,'ensure_target',decode)
    with pytest.raises(ValueError,match='during decoding'):
        study.checked_target(item,(1,1))


def test_direct_cli_environment_fails_closed(monkeypatch):
    for name in ('PYTHONHASHSEED','PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH'):
        monkeypatch.setenv(name,'0')
    assert study.require_environment_flags()['PYTHONHASHSEED']=='0'
    monkeypatch.setenv('PYTHONHASHSEED','1')
    with pytest.raises(ValueError,match='seed'):
        study.require_environment_flags()


def test_phase_identity_rejects_changed_parity(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'BASE',tmp_path)
    (tmp_path/'plan').mkdir();(tmp_path/'parity').mkdir()
    plan=tmp_path/'plan/plan.json';plan.write_text('plan')
    parity=tmp_path/'parity/parity.json';parity.write_text('parity')
    (tmp_path/'plan.log').write_text('plan log');(tmp_path/'parity.log').write_text('parity log')
    phases=[{'phase':name,'passed':True,'log_sha256':digest(tmp_path/(name+'.log')),
             'artifact_sha256':digest(path)} for name,path in [('plan',plan),('parity',parity)]]
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in phases))
    assert study.phase_inputs(tmp_path,plan,parity)==[digest(plan),digest(parity)]
    parity.write_text('changed')
    with pytest.raises(ValueError,match='parity artifact'):
        study.phase_inputs(tmp_path,plan,parity)


def test_independent_metric_agrees_on_present_and_empty_examples():
    import numpy as np
    from scripts.audit_boundary_video_factorial_v1 import metrics as independent
    from scripts.score_source_group_boundary_v1 import metric as primary
    for has_truth in (False,True):
        truth=np.zeros((25,25),dtype=bool);pred=truth.copy()
        if has_truth:truth[3:15,4:17]=True
        pred[5:14,7:20]=True
        x=independent(pred,truth);x.pop('present');x.pop('iou')
        x['zero_overlap']=(x['dice']==0) if truth.any() else None
        assert x==primary(pred,truth)


def test_factorial_reports_separate_matrix_contrast_and_interaction():
    row={'present':True,'global':{'dice':.4},'global_contrast':{'dice':.5},
         'foreground_background_no_contrast':{'dice':.6},'foreground_background':{'dice':.9},
         'foreground_background_asinh':{'dice':.8}}
    effects=study.factorial_effects([row])
    assert effects['matrix_without_contrast']==pytest.approx(.2)
    assert effects['matrix_with_contrast']==pytest.approx(.4)
    assert effects['contrast_with_global_matrix']==pytest.approx(.1)
    assert effects['contrast_with_region_matrix']==pytest.approx(.3)
    assert effects['interaction']==pytest.approx(.2)
    assert effects['asinh_head_input_vs_raw']==pytest.approx(-.1)
    assert study.factorial_effects([])['interaction'] is None


def test_frozen_scoring_and_independent_audit_on_synthetic_masks(tmp_path,monkeypatch):
    import numpy as np
    from PIL import Image
    monkeypatch.setattr(study,'outpath',lambda p:Path(p))
    monkeypatch.setattr(study,'deps',lambda:{'source':'frozen'})
    monkeypatch.setattr(study,'mask_path_sets',lambda *args:({'a'},{'b'}))
    monkeypatch.setattr(study,'phase_inputs',lambda *args:[digest(args[1]),digest(args[2])])
    plan_path=tmp_path/'plan.json';plan_path.write_text('plan')
    parity_path=tmp_path/'parity.json';parity_path.write_text('parity')
    plan={'environment_flags':{'seed':'0'}}
    monkeypatch.setattr(study,'require_plan',lambda *args:plan)
    monkeypatch.setattr(study,'require_parity',lambda *args:None)
    truth=np.zeros((20,20),dtype=np.uint8);truth[3:12,4:13]=1
    target=tmp_path/'target.png';Image.fromarray(truth).save(target)
    item={'split':'validation','sequence':'seq9','frame':4,'label':str(target),'label_sha256':digest(target)}
    pred_path=tmp_path/'pred.png';Image.fromarray(truth*255).save(pred_path)
    mask_row={'shape':[20,20],'baseline':str(pred_path),'refined':str(pred_path),
              'baseline_sha256':digest(pred_path),'refined_sha256':digest(pred_path)}
    runs={a:{'completion_sha256':a,'checkpoint_sha256':a,'frozen_sha256':a,
             'mapping':{('seq9',4):mask_row}} for a in study.ARMS}
    monkeypatch.setattr(study,'validated_runs',lambda *args:runs)
    proof={'passed':True,'manifest_sha256':'m','plan_sha256':digest(plan_path),
           'parity_sha256':digest(parity_path),'dependency_sha256':{'source':'frozen'},
           'environment_flags':plan['environment_flags'],'label_guard':{'open_attempts':0,'blocked_paths':2},
           'runs':{a:{k:r[k] for k in ('completion_sha256','checkpoint_sha256','frozen_sha256')} for a,r in runs.items()}}
    verify=tmp_path/'verification.json';verify.write_text(json.dumps(proof))
    (tmp_path/'replay.log').write_text('replayed')
    names=['plan','parity']+['train_'+a for a in study.ARMS]+['replay']
    phases=[{'phase':name,'passed':True} for name in names]
    phases[-1].update(log_sha256=digest(tmp_path/'replay.log'),verification_sha256=digest(verify))
    phase_file=tmp_path/'phases.jsonl';phase_file.write_text(''.join(json.dumps(p)+'\n' for p in phases))
    guard={'active':True,'open_attempts':0}
    study.score(tmp_path,{'samples':[item]},'m',plan_path,parity_path,verify,tmp_path/'scores',guard)
    assert not guard['active']
    (tmp_path/'score.log').write_text('scored')
    phases.append({'phase':'score','passed':True,'log_sha256':digest(tmp_path/'score.log'),
                   'artifact_sha256':digest(tmp_path/'scores/scores.json')})
    phase_file.write_text(''.join(json.dumps(p)+'\n' for p in phases))
    study.independent_audit(tmp_path,{'samples':[item]},'m',plan_path,parity_path,verify,
                            tmp_path/'audit',{'active':True,'open_attempts':0})
    assert json.loads((tmp_path/'audit/audit.json').read_text())['prediction_masks_checked']==1+len(study.ARMS)
    Image.fromarray(np.zeros_like(truth)).save(pred_path)
    with pytest.raises(ValueError,match='Prediction changed'):
        study.independent_audit(tmp_path,{'samples':[item]},'m',plan_path,parity_path,verify,
                                tmp_path/'audit2',{'active':True,'open_attempts':0})
