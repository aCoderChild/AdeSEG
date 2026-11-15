import torch
import torch.nn.functional as F

from modeling.cfbi_training import object_presence_loss


def test_author_presence_loss_supervises_absent_and_present_predictions():
    scores = torch.tensor([[-2.0], [2.0]], requires_grad=True)
    masks = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 0.0]]]])
    loss = object_presence_loss(scores, masks)
    expected = F.binary_cross_entropy_with_logits(scores, torch.tensor([[1.0], [0.0]]))
    assert torch.allclose(loss, expected)
    loss.backward()
    assert scores.grad[0, 0] < 0
    assert scores.grad[1, 0] > 0
