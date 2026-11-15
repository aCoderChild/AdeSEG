import pytest

from scripts.analyze_author_failure_sequences import group_ranges, metrics


def test_metrics_distinguishes_empty_predictions_disjoint_and_empty_gt():
    def row(present, dice, precision, recall, absent_fp=None):
        return {'present': present, 'scores': {'dice': dice, 'precision': precision,
                                               'recall': recall}, 'absent_fp': absent_fp}
    result = metrics([row(True, 0, 1, 0), row(True, 0, 0, 0),
                      row(True, 0.5, 0.5, 0.5), row(False, 0, 0, 1, True)])
    assert result['empty_predictions'] == 1
    assert result['disjoint_predictions'] == 1
    assert result['false_alarms'] == 1
    assert result['dice'] == pytest.approx(1 / 6)


def test_ranges_use_manifest_indices():
    assert group_ranges([1, 2, 5, 8, 9, 10]) == [[1, 2, 2], [5, 5, 1], [8, 10, 3]]
