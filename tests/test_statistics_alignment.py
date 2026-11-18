"""Statistical comparisons must not silently drop unmatched sequences."""
import pytest
from evaluation.statistics import paired_wilcoxon

def test_unmatched_sequence_sets_rejected():
    with pytest.raises(ValueError, match="sets differ"):
        paired_wilcoxon({"seq1": 0.5, "seq2": 0.6},
                        {"seq1": 0.7})

def test_equal_sequences_are_valid():
    result = paired_wilcoxon({"seq1": 0.5, "seq2": 0.6},
                             {"seq1": 0.5, "seq2": 0.6})
    assert result["n"] == 2
    assert result["p"] == 1.0
