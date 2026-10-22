"""Segmentation metrics and sequence-level statistical checks."""

import numpy as np
import pytest

from evaluation.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaluation.statistics import (
    failure_episodes,
    holm_adjust,
    rank_agreement,
    recovered_fraction,
    temporal_summary,
    paired_wilcoxon,
)


def test_retained_davis_metrics_score_identical_masks_as_perfect():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    assert db_eval_iou(mask, mask) == 1.0
    assert db_eval_boundary(mask, mask) == 1.0


def test_failure_episodes_are_maximal_and_require_three_frames():
    episodes = failure_episodes([0.8, 0.4, 0.2, 0.49, 0.8, 0.1, 0.2])
    assert [(item.start, item.end) for item in episodes] == [(1, 3)]
    summary = temporal_summary([0.8, 0.4, 0.2, 0.49, 0.8])
    assert summary["episode_count"] == 1
    assert summary["mean_episode_length"] == 3
    assert summary["recovery_rate"] == 1


def test_preregistered_statistics():
    assert recovered_fraction(0.4, 0.1, 0.6) == pytest.approx(0.5)
    agreement = rank_agreement([1, 2, 3], [10, 20, 30])
    assert agreement == {"spearman": 1.0, "kendall": 1.0}
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_wilcoxon_preserves_small_consistent_nonzero_differences():
    from scipy.stats import wilcoxon

    left = {str(i): 0.5 + i * 0.01 for i in range(10)}
    right = {k: v + 1e-6 for k, v in left.items()}
    result = paired_wilcoxon(left, right)
    assert result["p"] == pytest.approx(
        wilcoxon(list(left.values()), list(right.values())).pvalue
    )
    assert result["p"] < 0.01
    assert paired_wilcoxon(left, left)["p"] == 1
