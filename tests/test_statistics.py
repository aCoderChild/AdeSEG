"""Sequence-level reporting statistics, independent of model execution."""

import pytest

from evaluation.statistics import (
    failure_episodes,
    holm_adjust,
    rank_agreement,
    recovered_fraction,
    temporal_summary,
)


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
