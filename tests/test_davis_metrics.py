import numpy as np

from evaluation.davis2017.metrics import db_eval_boundary, db_eval_iou


def test_retained_davis_metrics_score_identical_masks_as_perfect():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    assert db_eval_iou(mask, mask) == 1.0
    assert db_eval_boundary(mask, mask) == 1.0
