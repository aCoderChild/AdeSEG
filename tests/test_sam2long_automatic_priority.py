import json

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.memory_token_io import digest
from scripts import verify_sam2long_automatic_gate as verify
from scripts import score_sam2long_automatic_priority as scorer
from scripts.report_sam2long_automatic_priority import metric,summarize


def test_gate_requires_complete_offline_clip_and_same_frozen_prompt(tmp_path):
    paths=[tmp_path/x for x in ('cpu','mps_a','mps_b')];sam=tmp_path/'samurai'
    for path in (*paths,sam):path.mkdir()
    common={'checkpoint_sha256':'checkpoint','config_sha256':digest(verify.CONFIG),
            'runner_sha256':digest(verify.ROOT/'scripts/run_sam2long_automatic_priority.py'),
            'manifest_sha256':digest(verify.MANIFEST),'source_revision':'rev','torch':'torch',
            'platform':'platform','machine':'arm64','seed':20261021,'dtype':'float32',
            'fallback':'0','fast_math':'0','pathways':verify.PATHWAY_CONFIG,'dependencies':verify.dependency_hashes()}
    prompt={'prompt_mask_sha256':'original'}
    for path,device in zip(paths,('cpu','mps','mps'),strict=True):
        records=[]
        for i in range(5):
            image=path/f'{i:06d}.png';Image.fromarray(np.zeros((2,3),dtype=np.uint8)).save(image)
            records.append({'frame':i,'image_sha256':str(i),'mask_sha256':digest(image)})
        receipt={'passed':True,'noncausal':True,'full_video':False,'deterministic':True,
                 'sequence':'seq20','prompt_frame':1,'prompt':prompt,'end_frame':4,
                 'pathway_audit':{'audited_post_prompt_frames':3,'pathways_after_initial_step':3,
                                  'checked_tensor_fields':['pred_masks','maskmem_features','obj_ptr','object_score_logits','ious']},
                 'identity':{**common,'device':device},'state_sha256':'state','records':records}
        (path/'receipt.json').write_text(json.dumps(receipt))
    (sam/'receipt.json').write_text(json.dumps({'passed':True,'full_video':True,'sequence':'seq20',
        'prompt_frame':1,'prompt':prompt,'checkpoint_sha256':'checkpoint'}))
    result=verify.audit(*paths,sam)
    assert result['passed'] and sum(row['mps_repeat_pixels'] for row in result['rows'])==0
    broken=json.loads((paths[2]/'receipt.json').read_text());broken['prompt']['prompt_mask_sha256']='other'
    (paths[2]/'receipt.json').write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='prompt'):verify.audit(*paths,sam)


def test_full_runner_rejects_failed_gate(tmp_path):
    # Bounds and hash checks must fail before any score is calculated.
    from scripts.run_sam2long_automatic_priority import verify_gate
    gate=tmp_path/'gate.json';gate.write_text(json.dumps({'passed':False}))
    with pytest.raises(ValueError,match='offline short-clip'):
        verify_gate(gate,'seq20',{}, {})


def test_frozen_offline_scorer_checks_same_inputs_and_pre_prompt(tmp_path):
    from types import SimpleNamespace
    lean,long=[tmp_path/x for x in ('lean','long')]
    lean.mkdir();long.mkdir();prompt={'mask':'same'}
    li={'device':'mps','fallback':'0','fast_math':'0','torch':'v','checkpoint_sha256':'weights',
        'runner_sha256':digest(verify.ROOT/'scripts/run_lean_window_baseline_mps.py')}
    xi={'device':'mps','fallback':'0','fast_math':'0','torch':'v','checkpoint_sha256':'weights',
        'pathways':verify.PATHWAY_CONFIG,'runner_sha256':digest(verify.ROOT/'scripts/run_sam2long_automatic_priority.py'),
        'config_sha256':digest(verify.CONFIG),'dependencies':verify.dependency_hashes()}
    lg={'passed':True,'sequence':'seq20','prompt':prompt,'identity':li,'state_sha256':'lean-state',
        'verifier_sha256':digest(verify.ROOT/'scripts/verify_lean_window_baseline_mps_gate.py')}
    xg={'passed':True,'sequence':'seq20','prompt':prompt,'identity':xi,'state_sha256':'long-state',
        'verifier_sha256':digest(verify.ROOT/'scripts/verify_sam2long_automatic_gate.py')}
    lp=tmp_path/'lean_gate.json';xp=tmp_path/'long_gate.json';lp.write_text(json.dumps(lg));xp.write_text(json.dumps(xg))
    frames=[];lr=[];xr=[]
    for i in range(3):
        image=tmp_path/f'image{i}.png';Image.fromarray(np.zeros((2,3),dtype=np.uint8)).save(image)
        frames.append(SimpleNamespace(image_path=image,frame_index=i))
        for directory,rows in ((lean,lr),(long,xr)):
            path=directory/f'{i:06d}.png';Image.fromarray(np.zeros((2,3),dtype=np.uint8)).save(path)
            rows.append({'frame':i,'source_frame_index':i,'image_sha256':digest(image),'mask_sha256':digest(path)})
    (lean/'receipt.json').write_text(json.dumps({'passed':True,'full_video':True,'sequence':'seq20','prompt_frame':1,
        'prompt':prompt,'identity':li,'gate_sha256':digest(lp),'state_sha256':'lean-state','end_frame':2,'records':lr}))
    (long/'receipt.json').write_text(json.dumps({'passed':True,'full_video':True,'noncausal':True,'sequence':'seq20','prompt_frame':1,
        'prompt':prompt,'identity':xi,'gate_sha256':digest(xp),'state_sha256':'long-state','end_frame':2,'records':xr,
        'pathway_audit':{'audited_post_prompt_frames':1,'pathways_after_initial_step':3,
                         'checked_tensor_fields':['pred_masks','maskmem_features','obj_ptr','object_score_logits','ious']}}))
    assert scorer.check_frozen('seq20',frames,lean,long,lp,xp,prompt)==1
    Image.fromarray(np.ones((2,3),dtype=np.uint8)).save(long/'000002.png')
    with pytest.raises(ValueError,match='frozen prediction changed'):
        scorer.check_frozen('seq20',frames,lean,long,lp,xp,prompt)


def test_pathway_audit_detects_missing_nonfinite_and_wrong_path_count():
    from scripts.run_sam2long_automatic_priority import audit_pathway_state
    def entry(paths):
        return {**{k:torch.zeros(1,paths) for k in ('pred_masks','maskmem_features','obj_ptr','object_score_logits','ious')},
                'acc_score':[0.,0.,0.]}
    state={'num_pathway':3,'output_dict':{'non_cond_frame_outputs':{2:entry(1),3:entry(3)}}}
    assert audit_pathway_state(state,1,3)['audited_post_prompt_frames']==2
    state['output_dict']['non_cond_frame_outputs'][3]['ious'][0,0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite ious'):audit_pathway_state(state,1,3)
    state['output_dict']['non_cond_frame_outputs'][3]=entry(1)
    with pytest.raises(ValueError,match='pred_masks'):audit_pathway_state(state,1,3)


def test_report_excludes_prompt_and_preserves_false_alarm_denominator():
    empty=np.zeros((2,3),dtype=bool);one=empty.copy();one[0,0]=True
    rows=[{'post_prompt':False,'native':metric(one,one)},
          {'post_prompt':True,'native':metric(empty,one)},
          {'post_prompt':True,'native':metric(one,empty)}]
    assert summarize(rows,'native')=={'present_frames':1,'mean_present_dice':0.,
        'empty_frames':1,'false_alarms':1,'mean_empty_predicted_pixels':1.}
