import torch

from modeling.region_contrast import RegionContrastBank, author_region_center


def _regions():
    labels = torch.zeros(64, 64, dtype=torch.long)
    labels[:, :21] = 1
    labels[:, 21:42] = 2
    return labels


def test_official_region_center_constructor_returns_three_roles():
    features = torch.randn(1, 256, 64, 64, requires_grad=True)
    labels = _regions()
    pred = torch.nn.functional.one_hot(labels, 3).permute(2, 0, 1)[None].float()
    centers, classes = author_region_center()(None, features, pred)
    assert centers.shape == (3, 256)
    assert classes.tolist() == [0, 1, 2]
    centers.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_regioncontrast_loss_updates_role_queues_and_backpropagates_to_q_and_k():
    torch.manual_seed(13)
    bank = RegionContrastBank(capacity=8, temperature=0.2)
    query = torch.randn(1, 4096, 256, requires_grad=True)
    memory = torch.randn(2 * 4096, 1, 256, requires_grad=True)
    labels = torch.cat((_regions().flatten(), _regions().flatten()))

    loss, info = bank(query, memory, labels, _regions())

    assert torch.isfinite(loss)
    assert info["query_roles"] == [0, 1, 2]
    assert info["contrast_roles"] == 3
    assert bank.count.tolist() == [2, 2, 2]
    loss.backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert memory.grad is not None and torch.isfinite(memory.grad).all()
    assert query.grad.abs().sum() > 0
    assert memory.grad.abs().sum() > 0


def test_regioncontrast_skips_without_negative_role_and_keeps_finite_zero_loss():
    bank = RegionContrastBank(capacity=4)
    query = torch.ones(1, 4096, 256, requires_grad=True)
    memory = torch.ones(4096, 1, 256, requires_grad=True)
    labels = torch.zeros(64 * 64, dtype=torch.long)

    loss, info = bank(query, memory, labels, torch.zeros(64, 64, dtype=torch.long))

    assert loss.item() == 0
    assert info["contrast_roles"] == 0
    loss.backward()
    assert query.grad is not None
    assert memory.grad is None or torch.isfinite(memory.grad).all()
