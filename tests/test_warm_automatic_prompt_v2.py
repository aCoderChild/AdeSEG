import numpy as np
import pytest
from modeling.warm_automatic_prompt_v2 import apply_archived_warm_prompt_v2
import torch


class Predictor:
    def __init__(self,change=False):self.calls=[];self.change=change
    def add_new_points_or_box(self,state,**kwargs):
        self.calls.append(('points_or_box',state,kwargs));value=torch.ones(1,1,4,5)
        if self.change and 'points' in kwargs:value[...,0,0]=-1
        return 0,[1],value
    def add_new_mask(self,state,**kwargs):
        self.calls.append(('mask',state,kwargs));return 0,[1],None


def test_mismatching_generated_mask_commits_archived_and_uses_it_for_prototypes():
    model=Predictor(change=True);state={};archived=np.ones((4,5),bool);seen=[]
    out=apply_archived_warm_prompt_v2(model,state,2,[0,0,5,4],[[1,1],[2,2]],archived,
        before_mask=lambda mask:seen.append(mask.copy()))
    assert not out['generated_equals_archived'] and out['different_pixels']==1
    assert np.array_equal(seen[0],archived) and np.array_equal(model.calls[-1][2]['mask'],archived)
    assert len(model.calls)==3 and all(call[1] is state for call in model.calls)


def test_mutating_callback_cannot_change_fixed_archived_mask():
    model=Predictor()
    def mutate(mask):mask[0,0]=False
    with pytest.raises(ValueError,match='changed before commit'):
        apply_archived_warm_prompt_v2(model,{},2,[0,0,5,4],[[1,1],[2,2]],np.ones((4,5),bool),before_mask=mutate)
    assert len(model.calls)==2


def test_v1_and_v2_match_when_generated_mask_is_archived():
    from modeling.warm_automatic_prompt_v1 import apply_archived_warm_prompt
    mask=np.ones((4,5),bool);a=Predictor();b=Predictor();seen_a=[];seen_b=[]
    result_a=apply_archived_warm_prompt(a,{},2,[0,0,5,4],[[1,1],[2,2]],mask,
        before_mask=lambda value:seen_a.append(value.copy()))
    result_b=apply_archived_warm_prompt_v2(b,{},2,[0,0,5,4],[[1,1],[2,2]],mask,
        before_mask=lambda value:seen_b.append(value.copy()))
    assert result_b['generated_equals_archived'] and result_b['different_pixels']==0
    assert np.array_equal(result_a,result_b['conditioning_mask']) and np.array_equal(seen_a[0],seen_b[0])
    assert np.array_equal(a.calls[-1][2]['mask'],b.calls[-1][2]['mask'])
