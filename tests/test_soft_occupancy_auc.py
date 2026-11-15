import pytest
from evaluation.soft_occupancy_auc import soft_occupancy_auc


def test_binary_auc_direction_and_ties():
    assert soft_occupancy_auc([0,1],[0,1])==1
    assert soft_occupancy_auc([1,0],[0,1])==0
    assert soft_occupancy_auc([.5,.5],[0,1])==.5
    assert soft_occupancy_auc([0,1],[0,0]) is None


def test_soft_occupancy_matches_explicit_weighted_pairs():
    scores=[.1,.8,.8];y=[.2,.7,.9];numerator=0
    for s,p in zip(scores,y):
        for t,q in zip(scores,y):numerator+=p*(1-q)*(1 if s>t else .5 if s==t else 0)
    expected=numerator/(sum(y)*sum(1-x for x in y))
    assert soft_occupancy_auc(scores,y)==pytest.approx(expected)
    with pytest.raises(ValueError):soft_occupancy_auc([0,1],[0,2])
