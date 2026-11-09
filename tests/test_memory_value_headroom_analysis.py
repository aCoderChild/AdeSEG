import json
import pytest
import torch

from scripts.analyze_memory_value_headroom import ARMS, summarize
from scripts.run_memory_value_headroom import load_case, score
from scripts.memory_token_io import digest


def case(name, frame, gain, present=True):
    return {'id':f'{name}_{frame}','sequence':name,'present':present,'arms':{
        a:{'scores':{'dice':.5+(gain if a in ('mask_only','mask_and_presence') else 0),
                    'iou':.4,'boundary_f':.3},'prediction_nonempty':present} for a in ARMS}}


def test_equal_sequence_weighting_and_empty_targets_excluded_from_dice():
    rows=[case('a',i,.1) for i in range(10)]+[case('b',0,-.1),case('b',1,0,False)]
    result=summarize(rows,7)
    assert result['contrasts']['mask_only']['mean_delta_dice']==pytest.approx(0)
    assert result['empty_target_count']==1
    assert result['sequences']['b']['native']['present_targets']==1
    assert not result['primary_headroom_screen_passed']
    assert 'not_estimable' in result['redetection_assessment']


def test_one_outlier_does_not_pass_predeclared_headroom_screen():
    result=summarize([case(str(i),0,.4 if i==0 else 0) for i in range(10)],7)
    assert result['contrasts']['mask_only']['mean_delta_dice']>.01
    assert not result['primary_headroom_screen_passed']
    assert not result['primary_decision_conditions']['all_leave_one_out_positive']


def test_positive_primary_with_no_empty_target_requires_later_fa_pilot():
    result=summarize([case(str(i),0,.05) for i in range(10)],7)
    assert result['primary_headroom_screen_passed']
    assert result['false_alarm_assessment']=='unavailable_require_later_false_alarm_pilot'
    assert not result['efficacy_authorized']
    rows=[case(str(i),0,.05) for i in range(10)] + [case('0',1,0,False)]
    rows[-1]['arms']['mask_only']['prediction_nonempty']=True
    assert not summarize(rows,7)['primary_headroom_screen_passed']


def test_saved_case_rejects_unreceipted_or_altered_result(tmp_path):
    out=tmp_path;(out/'cases').mkdir()
    assert load_case(out,'a','identity') is None
    path=out/'cases/a.json';path.write_text(json.dumps({'id':'a','identity_sha256':'identity'}))
    with pytest.raises(RuntimeError):load_case(out,'a','identity')
    (out/'cases/a.done.json').write_text(json.dumps({'identity_sha256':'identity','result_sha256':digest(path)}))
    assert load_case(out,'a','identity')['id']=='a'
    with pytest.raises(RuntimeError):load_case(out,'a','another-identity')
    path.write_text('{}')
    with pytest.raises(RuntimeError):load_case(out,'a','identity')


def test_score_uses_native_zero_logit_binary_convention():
    import numpy as np
    truth=np.ones((8,8),dtype=bool)
    output={'pred_masks':torch.ones(1,1,4,4),'object_score_logits':torch.ones(1,1)}
    assert score(output,truth)['scores']['dice']==1
    output['pred_masks'].zero_()
    assert score(output,truth)['scores']['dice']==0
    assert not score(output,truth)['prediction_nonempty']
