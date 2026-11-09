import copy
import pytest
from scripts.rde_value_results import summarize,ARMS


def groups():
    def row(d):return {'target_frames':12,'present_frames':10,'gt_empty_frames':2,'prompt_frame':0,'present_dice':d,
        'present_iou':d,'present_boundary_f':d,'gt_empty_false_alarm_rate':.5,'gt_empty_false_alarm_count':1,
        'elapsed_seconds':2.,'propagation':{'returns':[{'return_frame':5,'followup_sampled_frames':3,'detected':True,'delay_sampled_frames':0}]}}
    return {a:{n:{'sequence':n,**row(d)} for n in ['seq13','seq14','seq15']} for a,d in zip(ARMS,[.8,.805,.811])}


def test_new_matched_threshold_and_two_primary_contrasts():
    result=summarize(groups(),['seq13','seq14','seq15'],20261109,True)
    assert result['validation_screen']['passed']
    assert result['primary_contrasts_C2_minus']['native']['mean_delta']==pytest.approx(.011)
    assert result['primary_contrasts_C2_minus']['segmentation_only']['mean_delta']==pytest.approx(.006)
    assert len(result['primary_contrasts_C2_minus'])==2
    assert result['temporal_and_compute']['native']['gt_empty_frames']==6


def test_frame_coverage_and_return_pairing_must_match():
    g=groups();g['segmentation_only']['seq13']['target_frames']=11
    with pytest.raises(ValueError,match='denominators'):summarize(g,['seq13','seq14','seq15'],1)
    g['segmentation_only']['seq13']['present_frames']=9
    with pytest.raises(ValueError,match='schedule'):summarize(g,['seq13','seq14','seq15'],1)
    g=groups();g['segmentation_only']['seq13']['propagation']['returns'][0]['return_frame']=6
    with pytest.raises(ValueError,match='return events'):summarize(g,['seq13','seq14','seq15'],1)


def test_failed_screen_cannot_open_a_different_cohort_or_ignore_absence_safety():
    g=groups()
    for row in g['segmentation_plus_consistency'].values():row['gt_empty_false_alarm_rate']=.6
    r=summarize(g,['seq13','seq14','seq15'],1,True)
    assert not r['validation_screen']['passed']
    assert not r['validation_screen']['conditions']['false_alarm_increase_at_most_0_02']
    with pytest.raises(ValueError):summarize(g,['seq13','seq14'],1,True)
