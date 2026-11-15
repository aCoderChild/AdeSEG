"""Threshold-free weighted ROC AUC for soft spatial occupancy diagnostics."""
import numpy as np


def weighted_rank_auc(scores, positive_weights, negative_weights):
    """Probability a positive scores above a negative, with ties worth one half."""
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    pos = np.asarray(positive_weights, dtype=np.float64).reshape(-1)
    neg = np.asarray(negative_weights, dtype=np.float64).reshape(-1)
    if score.shape != pos.shape or score.shape != neg.shape:
        raise ValueError('Scores and weights must have equal shape')
    if not (np.isfinite(score).all() and np.isfinite(pos).all() and np.isfinite(neg).all()):
        raise ValueError('Scores and weights must be finite')
    if (pos < 0).any() or (neg < 0).any():
        raise ValueError('Weights must be nonnegative')
    total_pos, total_neg = pos.sum(), neg.sum()
    if total_pos == 0 or total_neg == 0:
        return None
    order = np.argsort(score, kind='stable')
    score, pos, neg = score[order], pos[order], neg[order]
    first = np.r_[0, np.flatnonzero(score[1:] != score[:-1]) + 1]
    group_pos = np.add.reduceat(pos, first)
    group_neg = np.add.reduceat(neg, first)
    negative_before = np.cumsum(group_neg) - group_neg
    return float(np.sum(group_pos * (negative_before + 0.5 * group_neg)) / (total_pos * total_neg))
