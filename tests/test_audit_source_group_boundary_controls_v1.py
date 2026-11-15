import numpy as np
import pytest

from scripts.audit_source_group_boundary_controls_v1 import ARMS, metrics, matched_contrasts, summarize
from scripts.score_source_group_boundary_controls_v1 import metric, summary


@pytest.mark.parametrize('case', ['empty', 'one_empty', 'touch_edge', 'perturbed', 'full'])
def test_independent_boundary_metrics(case):
    truth = np.zeros((43, 48), dtype=bool)
    pred = np.zeros_like(truth)
    if case == 'one_empty':
        truth[14:24, 12:28] = True
    elif case == 'touch_edge':
        truth[:25, :14] = True; pred[:23, 2:16] = True
    elif case == 'perturbed':
        truth[5:30, 9:34] = True; pred[7:32, 11:36] = True
    elif case == 'full':
        truth.fill(True); pred.fill(True)
    assert metrics(pred, truth) == pytest.approx(metric(pred, truth))


def test_independent_summary_and_reference_contrast():
    truth = np.zeros((32, 32), dtype=bool); truth[8:15, 7:16] = True
    empty = np.zeros_like(truth)
    pred = np.zeros_like(truth); pred[9:16, 7:16] = True
    rows = [{'present': True, 'arm': metric(pred, truth)},
            {'present': False, 'arm': metric(pred, empty)}]
    assert summarize(rows, 'arm') == pytest.approx(summary(rows, 'arm'))
    ref = {'pooled_frames': {'mean_present_dice': .6, 'mean_present_boundary_f_5px': .5, 'empty_false_alarms': 2},
           'equal_video_mean_present_dice': .4}
    other = {'pooled_frames': {'mean_present_dice': .7, 'mean_present_boundary_f_5px': .8, 'empty_false_alarms': 1},
             'equal_video_mean_present_dice': .6}
    data = {arm: ref if arm == 'region_boundary' else other for arm in ARMS}
    result = matched_contrasts(data)
    assert result['region_boundary']['pooled_present_dice_delta'] == 0
    for arm in ARMS[1:]:
        assert result[arm]['pooled_present_dice_delta'] == pytest.approx(.1)
        assert result[arm]['pooled_boundary_f_delta'] == pytest.approx(.3)
        assert result[arm]['empty_false_alarm_delta'] == -1
