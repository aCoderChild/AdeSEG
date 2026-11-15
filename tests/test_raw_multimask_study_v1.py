import pytest
import torch
from scripts.raw_multimask_shared_v1 import ARMS, LOSSES, model_for, dependency_hashes
from scripts.train_region_mixed_loss_v1 import state_sha256, pixel_loss
from scripts.score_raw_multimask_study_v1 import validate_training_order


def test_matched_capacity_and_initial_state():
    states=[]
    for arm in ARMS:
        torch.manual_seed(20261114)
        model=model_for(arm)
        assert sum(p.numel() for p in model.parameters())==48453
        states.append(state_sha256(model))
    assert len(set(states))==1
    assert LOSSES==('bce','robust_bce')
    with pytest.raises(ValueError):model_for('bad')


def test_loss_finite_for_native_suppressed_logits_and_empty():
    for kind in LOSSES:
        for target_value in (0.,1.):
            prediction=torch.full((1,1,2,2),-1024.,requires_grad=True)
            loss=pixel_loss(prediction,torch.full_like(prediction,target_value),kind)
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()


def test_source_provenance_and_incomplete_history():
    deps=dependency_hashes()
    for path in ('scripts/raw_multimask_shared_v1.py','scripts/train_raw_multimask_study_v1.py','scripts/score_raw_multimask_study_v1.py','modeling/raw_multimask_refiner.py'):
        assert len(deps[path])==64
    with pytest.raises(ValueError):validate_training_order([],{'samples':[]})
