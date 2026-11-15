import numpy as np
import pytest

from scripts.audit_source_group_refiner_scores_v1 import metrics, same, summarize


def test_empty_label_and_prediction_distinguish_false_alarm():
    empty = np.zeros((2, 2), dtype=bool)
    prediction = empty.copy(); prediction[0, 0] = True
    assert metrics(empty, empty)['dice'] == 1.0
    result = metrics(prediction, empty)
    assert result['dice'] == 0 and result['false_alarm'] is True
    assert result['precision'] == 0 and result['recall'] is None


def test_per_video_aggregation_excludes_undefined_precision():
    absent_prediction = {'dice': 0., 'precision': None, 'recall': 0., 'predicted_pixels': 0, 'false_alarm': None}
    correct_prediction = {'dice': 1., 'precision': 1., 'recall': 1., 'predicted_pixels': 2, 'false_alarm': None}
    summary = summarize([{'present': True, 'arm': absent_prediction},
                         {'present': True, 'arm': correct_prediction}], 'arm')
    assert summary['mean_present_dice'] == 0.5
    assert summary['mean_present_precision_when_defined'] == 1.
    assert summary['precision_defined_frames'] == 1
    assert summary['empty_frames'] == 0 and summary['mean_empty_predicted_pixels'] is None


def test_shape_mismatch_and_null_comparison():
    with pytest.raises(ValueError, match='geometries'):
        metrics(np.zeros((1, 2), dtype=bool), np.zeros((2, 1), dtype=bool))
    assert same(None, None) and not same(None, 0)
