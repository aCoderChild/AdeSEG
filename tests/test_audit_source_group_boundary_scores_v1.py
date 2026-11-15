import numpy as np
import pytest

from scripts.audit_source_group_boundary_scores_v1 import metrics as independent_metric
from scripts.audit_source_group_boundary_scores_v1 import matched_contrasts
from scripts.audit_source_group_boundary_scores_v1 import summarize as independent_summary
from scripts.score_source_group_boundary_v1 import metric, summary


@pytest.mark.parametrize('case', ['empty', 'one_empty', 'touch_edge', 'perturbed', 'full'])
def test_independent_metric_matches_scorer_for_mask_geometries(case):
    rng = np.random.default_rng(17)
    truth = np.zeros((43, 48), dtype=bool)
    pred = np.zeros_like(truth)
    if case == 'one_empty':
        truth[14:24, 12:28] = True
    elif case == 'touch_edge':
        truth[0:25, 0:14] = True
        pred[0:23, 2:16] = True
    elif case == 'perturbed':
        truth[5:30, 9:34] = True
        pred[7:32, 11:36] = True
        pred[rng.integers(0, 43, 10), rng.integers(0, 48, 10)] = True
    elif case == 'full':
        truth.fill(True)
        pred.fill(True)
    assert independent_metric(pred, truth) == pytest.approx(metric(pred, truth))


def test_independent_summary_matches_scorer():
    truth = np.zeros((32, 32), dtype=bool); truth[8:15, 7:16] = True
    empty = np.zeros_like(truth)
    pred = np.zeros_like(truth); pred[9:16, 7:16] = True
    rows = [{'present': True, 'arm': metric(pred, truth)},
            {'present': False, 'arm': metric(pred, empty)}]
    assert independent_summary(rows, 'arm') == pytest.approx(summary(rows, 'arm'))


def test_matched_objective_contrasts_keep_pairing_and_direction():
    base = {'pooled_frames': {'mean_present_dice': .6, 'mean_present_boundary_f_5px': .5, 'empty_false_alarms': 2},
            'per_video': {'seq9': {'mean_present_dice': .4}}, 'equal_video_mean_present_dice': .4}
    improved = {'pooled_frames': {'mean_present_dice': .7, 'mean_present_boundary_f_5px': .8, 'empty_false_alarms': 1},
                'per_video': {'seq9': {'mean_present_dice': .6}}, 'equal_video_mean_present_dice': .6}
    data = {f'{m}_{o}': (base if o == 'bce' else improved)
            for m in ('region', 'no_contrast', 'mask_only') for o in ('bce', 'boundary')}
    for contrast in matched_contrasts(data).values():
        assert contrast['pooled_present_dice_delta'] == pytest.approx(.1)
        assert contrast['pooled_boundary_f_delta'] == pytest.approx(.3)
        assert contrast['empty_false_alarm_delta'] == -1
        assert contrast['per_video_present_dice_delta']['seq9'] == pytest.approx(.2)
