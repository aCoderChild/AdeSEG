import pytest
import torch
from torch import nn

from modeling.decoder_adaptation import assert_feature_cache_safe, decoder_loss, train_decoder_core


def test_present_raw_mask_receives_gradient_despite_negative_presence():
    mask = torch.zeros(1,1,4,4,requires_grad=True)
    score = torch.tensor([[-3.]],requires_grad=True)
    iou = torch.tensor([[.1]],requires_grad=True)
    target = torch.ones(1,1,8,8)
    loss, terms = decoder_loss([(mask,iou,None,score)],target,resolution=8)
    loss.backward()
    assert torch.isfinite(loss) and mask.grad.abs().sum()>0 and score.grad.abs().sum()>0
    assert iou.grad.abs().sum()>0 and terms['loss_mask']>0


def test_absent_target_uses_author_presence_loss_without_mask_gradient():
    mask = torch.zeros(1,1,4,4,requires_grad=True)
    score = torch.tensor([[3.]],requires_grad=True)
    iou = torch.tensor([[.1]],requires_grad=True)
    loss, terms = decoder_loss([(mask,iou,None,score)],torch.zeros(1,1,8,8),resolution=8)
    loss.backward()
    assert torch.isfinite(loss) and score.grad.abs().sum()>0
    assert terms['loss_mask']==0 and terms['loss_dice']==0


def test_core_training_freezes_cache_producers_and_rejects_stale_cache():
    model = nn.Module()
    model.image_encoder = nn.Linear(2,2)
    model.memory_attention = nn.Linear(2,2)
    model.sam_mask_decoder = nn.Module()
    for name in ['conv_s0','conv_s1','transformer']:
        setattr(model.sam_mask_decoder,name,nn.Linear(2,2))
    selected = train_decoder_core(model)
    assert {name for name,p in selected}=={'sam_mask_decoder.transformer.weight','sam_mask_decoder.transformer.bias'}
    model.image_encoder.train()
    with pytest.raises(ValueError,match='Cached image features'):
        assert_feature_cache_safe(model)
    model.image_encoder.eval()
    model.sam_mask_decoder.conv_s0.weight.requires_grad_(True)
    with pytest.raises(ValueError,match='Cached image features'):
        assert_feature_cache_safe(model)
