import pytest
import torch
from scripts.bounded_anchor_shared_v1 import ARMS,LOSSES,dependency_hashes,model_for,verify_anchor_audit
from scripts.train_region_mixed_loss_v1 import state_sha256
from scripts.score_bounded_anchor_study_v1 import validate_training_order


def test_capacity_seed_and_source_binding():
    states=[]
    for arm in ARMS:
        torch.manual_seed(20261114);model=model_for(arm)
        assert sum(p.numel() for p in model.parameters())==48453
        states.append(state_sha256(model))
    assert len(set(states))==1 and LOSSES==('bce','robust_bce')
    deps=dependency_hashes()
    for path in ('modeling/bounded_anchor_raw_refiner.py','scripts/bounded_anchor_shared_v1.py',
                 'scripts/train_bounded_anchor_study_v1.py','scripts/score_bounded_anchor_study_v1.py',
                 'scripts/audit_bounded_anchor_v1.py'):
        assert len(deps[path])==64
    with pytest.raises(ValueError):model_for('unknown')


def test_invalid_history_rejected():
    with pytest.raises(ValueError):validate_training_order([],{'samples':[]})
