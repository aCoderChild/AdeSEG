"""No split leakage, matched data plans and faithful stopping/optimizer rules."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_iterative_memory_pilot as pilot


def protocol():
    return json.loads((Path(__file__).resolve().parents[1]/pilot.PROTOCOL).read_text())


def test_frozen_split_and_training_coverage():
    p=protocol();data=p['data'];t=p['training']
    assert not(set(data['train'])&set(data['validation']) or set(data['train'])&set(data['test'])
               or set(data['validation'])&set(data['test']))
    assert len(data['train'])*t['clips_per_sequence_per_epoch']*t['epochs']==480
    assert t['clip_length']==8 and (t['clip_length']-1)*480==3360
    assert t['validation_epochs']==[3,6,9,12]
    assert t['checkpoint_every_clips']==10


def test_plan_uses_only_training_sequences_and_preserves_global_random(monkeypatch):
    p=protocol()
    videos={s:[SimpleNamespace(frame_index=i) for i in range(20)] for s in p['data']['train']}
    monkeypatch.setattr(pilot,'sample_clip',lambda frames,length,seed:frames[seed%10:seed%10+length])
    state=pilot.random.getstate()
    a=pilot.training_plan(videos,p);b=pilot.training_plan(videos,p)
    assert a==b and len(a)==480
    assert pilot.random.getstate()==state
    assert {r['sequence'] for r in a}==set(p['data']['train'])
    assert all(r['indices']==list(range(r['indices'][0],r['indices'][0]+8)) for r in a)


def test_gradient_clipping_uses_pytorch_hook():
    module=torch.nn.Linear(2,1)
    optimizer=pilot.optimizer_for(module,protocol())
    for parameter in module.parameters():
        parameter.grad=torch.full_like(parameter,1000.)
    optimizer.step()
    norm=torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in module.parameters()]))
    assert norm<=3.00001 and norm>2.99


def summaries(gain=.02):
    records={}
    for arm in ('native',*pilot.ARMS):
        value=.7+gain if arm=='iterative_regional' else .7
        records[arm]={s:{'present_dice':value,'present_iou':.6,'present_boundary_f':.5,
                        'elapsed_seconds':1.,'gt_empty_false_alarm_count':1,'gt_empty_frames':10}
                      for s in protocol()['data']['test']}
    return records


def test_negative_result_cannot_advance():
    result=pilot.report(summaries(gain=0.),protocol())
    assert result['continuation_passed'] is False
    assert not result['continuation_checks']['native']


def test_matched_control_and_safety_are_required():
    records=summaries();result=pilot.report(records,protocol())
    assert result['continuation_passed'] is True
    for row in records['iterative_generic'].values():
        row['present_dice']=.72
    assert pilot.report(records,protocol())['continuation_passed'] is False
    records=summaries()
    records['iterative_regional']['seq16']['gt_empty_false_alarm_count']=10
    assert pilot.report(records,protocol())['continuation_checks']['false_alarm_safety'] is False


def test_missing_sequence_fails_closed():
    records=summaries();records['iterative_regional'].pop('seq20')
    with pytest.raises(RuntimeError,match='Incomplete'):
        pilot.report(records,protocol())
