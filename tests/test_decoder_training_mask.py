import pytest
import torch

from modeling.decoder_training_mask import decoder_training_mask, present_mask_loss, select_decoder_mask


def test_selects_per_sample_and_preserves_gradient():
    masks = torch.randn(2, 3, 4, 4, requires_grad=True)
    ious = torch.tensor([[0.1, 0.8, 0.2], [0.9, 0.2, 0.1]])
    selected = select_decoder_mask([(masks, ious)])
    assert torch.equal(selected[:, 0], masks[torch.arange(2), torch.tensor([1, 0])])
    selected.sum().backward()
    assert torch.equal(masks.grad.sum((2, 3)), torch.tensor([[0., 16., 0.], [16., 0., 0.]]))


def test_hook_removed_after_error():
    decoder = torch.nn.Identity()
    with pytest.raises(RuntimeError):
        with decoder_training_mask(decoder) as captured:
            decoder(torch.tensor(1))
            raise RuntimeError('stop')
    assert len(captured) == 1
    assert not decoder._forward_hooks


def test_mask_loss_only_supervises_present_objects():
    logits = torch.ones(2, 1, 2, 2, requires_grad=True)
    target = torch.zeros_like(logits)
    target[1, 0, 0, 0] = 1
    loss = present_mask_loss(logits, target, torch.nn.functional.binary_cross_entropy_with_logits)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0]) == 0
    assert torch.count_nonzero(logits.grad[1]) == 4


def test_empty_batch_mask_loss_has_zero_gradient():
    logits = torch.ones(1, 1, 2, 2, requires_grad=True)
    loss = present_mask_loss(logits, torch.zeros_like(logits), torch.nn.functional.binary_cross_entropy_with_logits)
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(logits.grad) == 0
