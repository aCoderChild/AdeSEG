import pytest
from scripts.audit_memory_value_headroom import check_scores


def row(dice,iou,nonempty=True,score=1):
    return {'scores':{'dice':dice,'iou':iou},'prediction_nonempty':nonempty,'object_score':score}


def test_inconsistent_metrics_fail():
    with pytest.raises(RuntimeError,match='algebra'):check_scores(row(.8,.8),True)
    with pytest.raises(RuntimeError):check_scores(row(float('nan'),0),True)
    with pytest.raises(RuntimeError,match='object score'):check_scores(row(0,0,False,float('nan')),True)


def test_native_absent_mask_rule_and_empty_dice_checked():
    with pytest.raises(RuntimeError,match='absent'):check_scores(row(1,1,True,-1),True)
    with pytest.raises(RuntimeError,match='Empty-mask'):check_scores(row(1,1,False,-1),True)
    check_scores(row(0,0,False,-1),True)
    check_scores(row(1,1,False,-1),False)


def test_nonempty_false_alarm_has_zero_empty_gt_dice():
    check_scores(row(0,0,True,1),False)
    with pytest.raises(RuntimeError,match='False-alarm'):check_scores(row(1,1,True,1),False)
