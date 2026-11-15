import pytest
import torch

from scripts.diagnose_lean_raw_decoder_train_reads import select_raw_mask, state_hash


def test_selection_matches_predicted_iou_for_each_batch():
    masks=torch.arange(2*3*2*2,dtype=torch.float32).reshape(2,3,2,2)
    ious=torch.tensor([[0.1,0.9,0.2],[0.7,0.3,0.1]])
    selected,index=select_raw_mask(masks,ious,True)
    assert index.tolist()==[1,0]
    assert torch.equal(selected[0,0],masks[0,1]) and torch.equal(selected[1,0],masks[1,0])


def test_single_selection_and_invalid_inputs():
    masks=torch.randn(1,1,2,2);ious=torch.tensor([[0.4]])
    selected,index=select_raw_mask(masks,ious,False)
    assert torch.equal(selected,masks) and index.tolist()==[0]
    with pytest.raises(ValueError):select_raw_mask(masks,torch.randn(1,3),False)
    with pytest.raises(ValueError):select_raw_mask(torch.randn(1,3,2,2),torch.randn(1,3),False)
    masks[0,0,0,0]=float('nan')
    with pytest.raises(ValueError):select_raw_mask(masks,ious,False)


def test_state_hash_includes_parameters_and_buffers():
    model=torch.nn.BatchNorm2d(2)
    original=state_hash(model)
    model.running_mean[0]=1
    assert state_hash(model)!=original
    changed=state_hash(model)
    with torch.no_grad():model.weight[0]+=1
    assert state_hash(model)!=changed
