from scripts.summarize_native_stages import outcome


def row(present,dice=0,iou=0,precision=1,absent_fp=None):
    return {'present':present,'scores':{'dice':dice,'iou':iou,'precision':precision},
            'absent_fp':absent_fp}


def test_summary_separates_empty_and_disjoint_positive_predictions():
    assert outcome(row(True))=='missed_or_lost'
    assert outcome(row(True,precision=0))=='disjoint_prediction'
    assert outcome(row(True,dice=.5,iou=1/3,precision=.5))=='overlapping_prediction'


def test_summary_keeps_false_alarms_outside_present_dice_categories():
    assert outcome(row(False,absent_fp=True))=='false_alarm'
    assert outcome(row(False,dice=1,iou=1,absent_fp=False))=='correct_absence'


def test_summary_honours_explicit_mask_occupancy():
    value=row(True)
    value['prediction_nonempty']=True
    assert outcome(value)=='disjoint_prediction'
