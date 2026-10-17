"""Preregistered statistics for utility-aware memory retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau, spearmanr, wilcoxon


@dataclass(frozen=True)
class FailureEpisode:
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start + 1


def failure_episodes(scores, threshold=0.5, minimum_length=3):
    """Return maximal runs below the fixed failure threshold."""
    scores = np.asarray(scores, dtype=float)
    episodes, start = [], None
    for index, failed in enumerate(scores < threshold):
        if failed and start is None:
            start = index
        if start is not None and (not failed or index == len(scores) - 1):
            end = index if failed else index - 1
            if end - start + 1 >= minimum_length:
                episodes.append(FailureEpisode(start, end))
            start = None
    return episodes


def recovered_fraction(oracle_gain, matched_random_gain, oracle_memory_gain):
    """E2 headroom recovered after removing best-of-N search advantage."""
    if oracle_memory_gain <= 0:
        return float("nan")
    return float((oracle_gain - matched_random_gain) / oracle_memory_gain)


def rank_agreement(criterion, utility):
    criterion, utility = np.asarray(criterion), np.asarray(utility)
    valid = np.isfinite(criterion) & np.isfinite(utility)
    if (
        valid.sum() < 2
        or np.unique(criterion[valid]).size < 2
        or np.unique(utility[valid]).size < 2
    ):
        return {"spearman": float("nan"), "kendall": float("nan")}
    return {
        "spearman": float(spearmanr(criterion[valid], utility[valid]).statistic),
        "kendall": float(kendalltau(criterion[valid], utility[valid]).statistic),
    }


def paired_wilcoxon(left_by_sequence, right_by_sequence):
    """Paired test over sequence means; frames are never treated as independent."""
    names = sorted(set(left_by_sequence) & set(right_by_sequence))
    left = np.array([left_by_sequence[name] for name in names], dtype=float)
    right = np.array([right_by_sequence[name] for name in names], dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2 or np.allclose(left[valid], right[valid]):
        return {"n": int(valid.sum()), "statistic": float("nan"), "p": 1.0}
    result = wilcoxon(left[valid], right[valid], alternative="two-sided")
    return {
        "n": int(valid.sum()),
        "statistic": float(result.statistic),
        "p": float(result.pvalue),
    }


def sequence_bootstrap_interval(
    values_by_sequence, seed=20261016, resamples=10_000, confidence=0.95
):
    values = np.asarray(list(values_by_sequence.values()), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for index in range(resamples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    alpha = (1 - confidence) / 2
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))]


def holm_adjust(p_values):
    """Holm family-wise correction, returned in the input mapping's order."""
    items = sorted(p_values.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    count = len(items)
    for rank, (name, p_value) in enumerate(items):
        running = max(running, min(1.0, (count - rank) * float(p_value)))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def temporal_summary(scores, threshold=0.5, minimum_length=3):
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    episodes = failure_episodes(scores, threshold, minimum_length)
    return {
        "failure_rate": float(np.mean(scores[valid] < threshold))
        if valid.any()
        else None,
        "episode_count": len(episodes),
        "mean_episode_length": float(np.mean([item.length for item in episodes]))
        if episodes
        else 0.0,
        "recovery_rate": float(
            np.mean(
                [
                    item.end < len(scores) - 1
                    and np.isfinite(scores[item.end + 1])
                    and scores[item.end + 1] >= threshold
                    for item in episodes
                ]
            )
        )
        if episodes
        else (1.0 if valid.any() else None),
    }
