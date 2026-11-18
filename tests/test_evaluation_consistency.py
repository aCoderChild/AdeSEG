"""Metric contracts: empty masks, label vocabulary, temporal alignment."""
import numpy as np
import pytest
from evaluation.segmentation import calculate_scores
from evaluation.temporal_consistency import adjacent_mask_metrics, undefined_temporal_metrics

def test_empty_binary_masks_are_perfect():
    x = np.zeros((4, 4), dtype=np.uint8)
    scores = calculate_scores(x, x)
    assert scores["dice"] == 1
    assert scores["iou"] == 1
    assert scores["precision"] == 1

def test_missed_object_has_zero_precision_and_dice():
    pred = np.zeros((4, 4), dtype=np.uint8)
    gt = np.ones((4, 4), dtype=np.uint8)
    scores = calculate_scores(pred, gt)
    assert scores["precision"] == 0
    assert scores["recall"] == 0
    assert scores["dice"] == 0

def test_unknown_semantic_label_rejected():
    pred = np.array([[0, 1], [3, 2]])
    gt = np.array([[0, 1], [0, 2]])
    with pytest.raises(ValueError, match="Unknown prediction class"):
        calculate_scores(pred, gt, class_labels=(1, 2))

def test_no_pixels_is_invalid():
    with pytest.raises(ValueError, match="empty image"):
        calculate_scores(np.empty((0, 0)), np.empty((0, 0)))

def test_temporal_identical_and_empty_cases():
    a = np.zeros((8, 8), dtype=np.uint8)
    assert adjacent_mask_metrics(a, a)["temporal_iou"] == 1
    assert adjacent_mask_metrics(a, a)["area_change_relative"] == 0
    assert np.isnan(adjacent_mask_metrics(a, a)["centroid_shift_normalized"])
    a[2:4, 3:5] = 1
    x = adjacent_mask_metrics(a, a)
    assert x["temporal_iou"] == 1
    assert x["area_change_relative"] == 0
    assert x["centroid_shift_normalized"] == 0

def test_temporal_motion_and_missing():
    a = np.zeros((8, 8), dtype=np.uint8)
    b = a.copy()
    a[0,0] = 1
    b[0,1] = 1
    x = adjacent_mask_metrics(a, b)
    assert x["temporal_iou"] == 0
    assert x["area_change_relative"] == 0
    assert x["centroid_shift_normalized"] == pytest.approx(1 / np.hypot(8,8))
    assert all(np.isnan(v) for v in undefined_temporal_metrics().values())

def test_temporal_wrong_geometry_rejected():
    with pytest.raises(ValueError, match="aligned"):
        adjacent_mask_metrics(np.zeros((3, 3)), np.zeros((4, 4)))
