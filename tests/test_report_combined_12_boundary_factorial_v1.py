import pytest

from scripts.report_combined_12_boundary_factorial_v1 import ARMS, EXPECTED, GROUPS, paired_analysis


def test_groups_partition_all_twelve_patients():
    assert len(EXPECTED) == 12
    assert set().union(*GROUPS.values()) == EXPECTED
    assert sum(map(len, GROUPS.values())) == len(EXPECTED)
    assert GROUPS['prior_development_two'] == {'seq6', 'seq10'}


def test_paired_bootstrap_uses_all_patients_and_is_deterministic():
    per = {s: {a: {'mean_present_dice': .5 if a == 'A' else .6} for a in ARMS} for s in EXPECTED}
    result=paired_analysis(per,replicates=100,seed=7)
    assert result == paired_analysis(per,replicates=100,seed=7)
    assert result['D']['mean_delta'] == pytest.approx(.1)
    assert result['D']['percentile_95_ci'] == pytest.approx([.1,.1])
    assert result['D']['improved_patients'] == 12
    assert set(result['D']['subgroups']['prior_development_two']['sequences']) == {'seq6','seq10'}
    del per['seq6']
    with pytest.raises(ValueError,match='twelve'): paired_analysis(per,replicates=100)


def test_bootstrap_refuses_too_few_replicates():
    with pytest.raises(ValueError,match='sufficient'): paired_analysis({},replicates=10)
