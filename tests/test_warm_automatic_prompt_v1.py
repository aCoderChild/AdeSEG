import numpy as np
import pytest
import torch
from modeling.warm_automatic_prompt_v1 import apply_archived_warm_prompt


class Predictor:
    def __init__(self,change=False):self.calls=[];self.change=change
    def add_new_points_or_box(self,state,**kwargs):
        self.calls.append(('points_or_box',state,kwargs))
        value=torch.ones(1,1,4,5)
        if self.change and 'points' in kwargs:value[...,0,0]=-1
        return 0,[1],value
    def add_new_mask(self,state,**kwargs):
        self.calls.append(('mask',state,kwargs));return 0,[1],None


def test_warm_prompt_runs_box_points_mask_in_same_state():
    model=Predictor();state={};expected=np.ones((4,5),bool)
    result=apply_archived_warm_prompt(model,state,2,[0,0,5,4],[[1,1],[2,2]],expected,
        before_mask=lambda generated:model.calls.append(('feature',state,{})))
    assert np.array_equal(result,expected) and len(model.calls)==4
    assert all(call[1] is state for call in model.calls)
    assert 'box' in model.calls[0][2] and model.calls[1][2]['clear_old_points'] is False
    assert model.calls[1][2]['labels'].tolist()==[1,0] and model.calls[2][0]=='feature' and model.calls[3][0]=='mask'


def test_warm_prompt_mismatch_never_writes_memory():
    model=Predictor(change=True)
    with pytest.raises(ValueError,match='differs from archived'):
        apply_archived_warm_prompt(model,{},2,[0,0,5,4],[[1,1],[2,2]],np.ones((4,5),bool))
    assert len(model.calls)==2


def test_viewport_bounds_fail_before_model():
    model=Predictor()
    with pytest.raises(ValueError,match='outside'):
        apply_archived_warm_prompt(model,{},2,[0,0,5,4],[[5,1],[2,2]],np.ones((4,5),bool))
    assert not model.calls


def test_mutating_callback_cannot_change_mask_before_commit():
    model=Predictor()
    def mutate(generated):generated[0,0]=False
    with pytest.raises(ValueError,match='changed before commit'):
        apply_archived_warm_prompt(model,{},2,[0,0,5,4],[[1,1],[2,2]],
                                   np.ones((4,5),bool),before_mask=mutate)
    assert len(model.calls)==2
