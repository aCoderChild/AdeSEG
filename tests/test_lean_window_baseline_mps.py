import json

import numpy as np
from PIL import Image
import pytest

from scripts.memory_token_io import digest
from scripts import verify_lean_window_baseline_mps_gate as verify
from scripts import score_lean_window_mps_vs_samurai as scorer
from scripts import audit_lean_window_baseline_mps_full as full_audit
from scripts.report_lean_mps_vs_samurai import independent_metric,independent_summary


def test_gate_checks_archived_prompt_frames_masks_and_budget(tmp_path):
    a,b,cpu,sam=[tmp_path/x for x in ('a','b','cpu','sam')]
    for p in (a,b,cpu,sam):p.mkdir()
    identity={'runner_sha256':digest(verify.ROOT/'scripts/run_lean_window_baseline_mps.py'),
              'device':'mps','fallback':'0','fast_math':'0','checkpoint_sha256':'checkpoint'}
    prompt={'prompt_mask_sha256':'archived'}
    records=[]; hashes=[]
    for i in range(4):
        for p in (a,b,cpu):Image.fromarray(np.zeros((2,3),dtype=np.uint8)).save(p/f'{i:06d}.png')
        hashes.append(digest(cpu/f'{i:06d}.png'))
        records.append({'frame':i,'image_sha256':f'image{i}','mask_sha256':digest(a/f'{i:06d}.png')})
    events=[{'stage':'admission_plan','evict_frames':[0]},{'stage':'after_write','total_tokens':16384,'insurance_tokens':3072}]
    receipt={'passed':True,'full_video':False,'sequence':'seq20','prompt_frame':2,'prompt':prompt,
             'end_frame':3,'identity':identity,'state_sha256':'state','records':records,
             'token_window_events':events,'selection':[]}
    for p in (a,b):(p/'receipt.json').write_text(json.dumps(receipt))
    (sam/'receipt.json').write_text(json.dumps({'passed':True,'full_video':True,'sequence':'seq20',
        'prompt_frame':2,'prompt':prompt,'checkpoint_sha256':'checkpoint'}))
    (cpu/'records.json').write_text(json.dumps({'mask_sha256':hashes}))
    (cpu/'completion.json').write_text(json.dumps({'passed':True,'records_sha256':digest(cpu/'records.json')}))
    result=verify.verify(a,b,cpu,sam)
    assert result['passed'] and sum(x['mps_repeat_pixels'] for x in result['rows'])==0
    bad=json.loads((b/'receipt.json').read_text());bad['prompt']['prompt_mask_sha256']='other'
    (b/'receipt.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='identities differ'):verify.verify(a,b,cpu,sam)
    (b/'receipt.json').write_text(json.dumps(receipt))
    bad=json.loads((b/'receipt.json').read_text());bad['token_window_events'][1]['total_tokens']=16385
    (b/'receipt.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='budget'):verify.verify(a,b,cpu,sam)


def test_same_device_scorer_checks_full_rebase_and_frozen_masks(tmp_path):
    from types import SimpleNamespace
    lean=tmp_path/'lean';sam=tmp_path/'sam';lean.mkdir();sam.mkdir()
    prompt={'prompt_mask_sha256':'archived'}
    identity={'device':'mps','fallback':'0','fast_math':'0','torch':'v','checkpoint_sha256':'weights',
              'runner_sha256':digest(verify.ROOT/'scripts/run_lean_window_baseline_mps.py')}
    lg={'passed':True,'sequence':'seq20','prompt':prompt,'identity':identity,'state_sha256':'lean-state',
        'verifier_sha256':digest(verify.ROOT/'scripts/verify_lean_window_baseline_mps_gate.py')}
    sg={'passed':True,'sequence':'seq20','prompt':prompt,'state_sha256':'sam-state',
        'verifier_sha256':digest(verify.ROOT/'scripts/verify_samurai_automatic_gate.py')}
    lp=tmp_path/'lean_gate.json';sp=tmp_path/'sam_gate.json';lp.write_text(json.dumps(lg));sp.write_text(json.dumps(sg))
    frames=[];lr=[];sr=[]
    for i in range(3):
        image=tmp_path/f'image{i}.png';Image.fromarray(np.zeros((2,3),dtype=np.uint8)).save(image)
        frames.append(SimpleNamespace(image_path=image,frame_index=i))
        mask=lean/f'{i:06d}.png';Image.fromarray(np.zeros((2,3),dtype=np.uint8)).save(mask)
        lr.append({'frame':i,'source_frame_index':i,'image_sha256':digest(image),'mask_sha256':digest(mask)})
        if i:
            mask=sam/f'{i:06d}.png';Image.fromarray(np.zeros((2,3),dtype=np.uint8)).save(mask)
            sr.append({'frame':i,'local_index':i-1,'source_frame_index':i,'image_sha256':digest(image),'mask_sha256':digest(mask)})
    l={'passed':True,'full_video':True,'sequence':'seq20','prompt_frame':1,'prompt':prompt,'identity':identity,
       'gate_sha256':digest(lp),'state_sha256':'lean-state','end_frame':2,'records':lr}
    s={'passed':True,'full_video':True,'sequence':'seq20','prompt_frame':1,'prompt':prompt,'device':'mps',
       'fallback':'0','fast_math':'0','torch':'v','checkpoint_sha256':'weights','gate_sha256':digest(sp),
       'runner_sha256':digest(verify.ROOT/'scripts/run_samurai_automatic_priority.py'),
       'kalman_adapter_sha256':digest(verify.ROOT/'modeling/samurai_mps_kalman.py'),
       'state_sha256':'sam-state','frames':2,'records':sr}
    (lean/'receipt.json').write_text(json.dumps(l));(sam/'receipt.json').write_text(json.dumps(s))
    assert scorer.check_frozen('seq20',frames,lean,sam,lp,sp,prompt)[0]==1
    Image.fromarray(np.ones((2,3),dtype=np.uint8)).save(lean/'000000.png')
    with pytest.raises(ValueError,match='mask changed'):
        scorer.check_frozen('seq20',frames,lean,sam,lp,sp,prompt)


def test_full_auditor_compares_every_frozen_mask_and_rejects_tamper(tmp_path):
    mps,cpu,sam=[tmp_path/x for x in ('mps','cpu','sam')]
    for p in (mps,cpu,sam):p.mkdir()
    identity={'device':'mps'};prompt={'mask':'archived'}
    gate={'passed':True,'sequence':'seq20','prompt':prompt,'identity':identity,'state_sha256':'state',
          'verifier_sha256':digest(verify.ROOT/'scripts/verify_lean_window_baseline_mps_gate.py')}
    gp=tmp_path/'gate.json';gp.write_text(json.dumps(gate));records=[];hashes=[]
    for i in range(4):
        for p in (mps,cpu):Image.fromarray(np.zeros((2,3),dtype=np.uint8)).save(p/f'{i:06d}.png')
        hashes.append(digest(cpu/f'{i:06d}.png'));records.append({'frame':i,'mask_sha256':digest(mps/f'{i:06d}.png')})
    (cpu/'records.json').write_text(json.dumps({'mask_sha256':hashes}))
    (cpu/'completion.json').write_text(json.dumps({'passed':True,'records_sha256':digest(cpu/'records.json')}))
    (sam/'receipt.json').write_text(json.dumps({'passed':True,'full_video':True,'sequence':'seq20','prompt':prompt,'prompt_frame':2}))
    receipt={'passed':True,'full_video':True,'sequence':'seq20','prompt':prompt,'identity':identity,
        'prompt_frame':2,'end_frame':3,'records':records,'gate_sha256':digest(gp),'state_sha256':'state',
        'token_window_events':[{'stage':'admission_plan','evict_frames':[1]},
                               {'stage':'after_write','total_tokens':16384,'insurance_tokens':3072}]}
    (mps/'receipt.json').write_text(json.dumps(receipt))
    result=full_audit.audit(mps,cpu,sam,gp)
    assert result['frames']==4 and result['cpu_mps_differing_pixels']==0 and result['evictions']==1
    Image.fromarray(np.ones((2,3),dtype=np.uint8)).save(cpu/'000003.png')
    with pytest.raises(ValueError,match='mask hash'):full_audit.audit(mps,cpu,sam,gp)


def test_independent_report_denominators_exclude_prompt_and_separate_empty():
    empty=np.zeros((2,3),dtype=bool);one=empty.copy();one[0,0]=True
    rows=[{'post_prompt':False,'lean':independent_metric(one,one)},
          {'post_prompt':True,'lean':independent_metric(one,empty)},
          {'post_prompt':True,'lean':independent_metric(empty,one)}]
    result=independent_summary(rows,'lean',True)
    assert result=={'frames':2,'present_frames':1,'mean_present_frame_dice':0.,
                    'empty_frames':1,'false_alarm_frames':1,'mean_empty_predicted_pixels':1.}
