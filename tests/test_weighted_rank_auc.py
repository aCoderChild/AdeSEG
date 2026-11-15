import numpy as np
import pytest

from evaluation.weighted_rank_auc import weighted_rank_auc


def test_weighted_auc_matches_explicit_pairs_with_ties():
    scores = np.array([0.1, 0.9, 0.9, 0.3])
    positive = np.array([0.1, 0.7, 0.3, 0.2])
    negative = np.array([0.9, 0.3, 0.7, 0.8])
    explicit = sum(positive[i] * negative[j] * ((scores[i] > scores[j]) + .5 * (scores[i] == scores[j]))
                   for i in range(4) for j in range(4)) / (positive.sum() * negative.sum())
    assert weighted_rank_auc(scores, positive, negative) == pytest.approx(explicit)


def test_no_positive_or_negative_is_undefined_and_bad_weights_fail():
    assert weighted_rank_auc([0, 1], [0, 0], [1, 1]) is None
    assert weighted_rank_auc([0, 1], [1, 1], [0, 0]) is None
    with pytest.raises(ValueError, match='nonnegative'):
        weighted_rank_auc([0], [-1], [1])
