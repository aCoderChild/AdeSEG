import pytest
from scripts.audit_combined_12_boundary_factorial_v1 import bootstrap, close


def test_bootstrap_is_deterministic_and_constant():
    a=bootstrap([.1]*12,replicates=100,seed=7)
    assert a==bootstrap([.1]*12,replicates=100,seed=7)
    assert a[0]==pytest.approx(.1) and a[1]==pytest.approx([.1,.1]) and a[2:]==(12,0)


def test_bootstrap_rejects_missing_patients():
    with pytest.raises(ValueError,match='sample mismatch'):
        bootstrap([.1]*11,replicates=100)


def test_close_handles_none():
    assert close(None,None) and not close(None,0) and not close(0,None)
