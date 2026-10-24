import numpy as np
import pytest

from scripts.region_sparse_statistics import deficit_parts, pack_history, token_counts


def test_exact_deficit_partition():
    gt = np.zeros((64, 80), bool)
    gt[15:45, 20:55] = True
    for pred in (gt, np.zeros_like(gt), np.ones_like(gt), np.roll(gt, 8, axis=0)):
        result = deficit_parts(pred, gt)
        assert sum(result[k] for k in ("complete_miss", "edge", "undersegmentation", "excess_prediction")) == pytest.approx(1-result["dice"])
    assert deficit_parts(np.zeros_like(gt), gt)["complete_miss"] == 1


def test_ring_disjoint_and_one_token():
    labels = np.zeros((64, 64), int)
    labels[32, 32] = 1
    counts = token_counts(labels)
    assert counts["near_background"] == 8
    assert counts["retained"] == 9
    labels[0, 0] = 2
    assert token_counts(labels)["boundary"] == 1


def test_empty_full_and_finite_capacity():
    assert token_counts(np.zeros((64, 64), int))["hypothetical_older_capacity"] is None
    assert token_counts(np.ones((64, 64), int))["hypothetical_older_capacity"] == 5
    assert pack_history([4096] * 10) == {"older_frames": 5, "older_tokens": 20480}
    assert pack_history([0] * 3) == {"older_frames": 3, "older_tokens": 0}
    assert pack_history([])["older_frames"] == 0
    with pytest.raises(ValueError):
        pack_history([-1])


def test_reject_invalid_or_absent_gt():
    with pytest.raises(ValueError):
        deficit_parts(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        token_counts(np.zeros((32, 32)))


def test_nonempty_disjoint_is_not_empty_miss():
    gt = np.zeros((64, 64), bool)
    gt[8:15, 8:15] = True
    pred = np.roll(gt, 32, axis=0)
    r = deficit_parts(pred, gt)
    assert r["zero_overlap_nonempty"]
    assert r["complete_miss"] == 0
    assert r["dice"] == 0


def test_recent_suffix_budget_not_mean_capacity():
    result = pack_history([1, 4000, 4000], budget=5000)
    assert result == {"older_frames": 1, "older_tokens": 4000}
    assert pack_history([4096], budget=4096)["older_frames"] == 1
