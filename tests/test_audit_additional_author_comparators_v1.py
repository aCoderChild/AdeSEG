import numpy as np

from scripts.audit_additional_author_comparators_v1 import compare
from scripts.audit_boundary_video_factorial_v1 import metrics
from scripts.score_boundary_video_factorial_v1 import metric


def test_nested_comparison_reports_real_differences():
    errors = []
    compare(errors, ['seq'], {'a': 1, 'nested': {'x': 2}}, {'a': 1, 'nested': {'x': 3}})
    assert len(errors) == 1 and errors[0][:3] == ['seq', 'nested', 'x']


def test_independent_metric_matches_primary_on_synthetic_boundaries():
    truth = np.zeros((25, 28), dtype=bool)
    pred = np.zeros_like(truth)
    truth[2:12, 3:15] = True
    pred[3:15, 5:18] = True
    errors = []
    compare(errors, ['fixture'], metrics(pred, truth), metric(pred, truth))
    assert errors == []
