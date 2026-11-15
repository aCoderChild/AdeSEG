import pytest
import torch

from modeling.regional_low_rank import RegionalLowRankHeads, RegionalLowRankLayer


@pytest.mark.parametrize("pointers", [0, 3])
def test_identity_preserves_features_and_upstream_gradient_once(pointers):
    torch.manual_seed(8)
    layer = RegionalLowRankLayer(channels=8, rank=2)
    memory = torch.randn(1, 7 + pointers, 8, requires_grad=True)
    roles = torch.rand(1, 7, 3).softmax(-1)
    actual = layer(memory, roles, pointers)
    assert torch.equal(actual, memory)
    actual.square().sum().backward()
    torch.testing.assert_close(memory.grad, 2 * memory.detach(), atol=0, rtol=0)
    for region in layer.regions:
        assert not region.weight.requires_grad
        assert region.lora_B.grad.abs().sum() > 0
        assert region.lora_A.grad.count_nonzero() == 0


def test_transform_depends_on_token_content_and_preserves_pointers():
    layer = RegionalLowRankLayer(channels=8, rank=2)
    with torch.no_grad():
        for i, region in enumerate(layer.regions):
            region.lora_A.fill_(1)
            region.lora_B.fill_(i + 1)
    memory = torch.ones(1, 5, 8)
    memory[:, 1] *= 2
    roles = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [.5, 0., .5]]])
    delta = layer(memory, roles, pointers=1) - memory
    assert torch.equal(delta[:, 1], 2 * delta[:, 0])
    assert torch.equal(delta[:, 2], 2 * delta[:, 0])
    assert torch.equal(delta[:, 3], 2 * delta[:, 0])
    assert delta[:, 4].count_nonzero() == 0


def test_author_eval_does_not_merge_or_change_adapter_output():
    layer = RegionalLowRankLayer(channels=8, rank=2)
    with torch.no_grad():
        for region in layer.regions:
            region.lora_B.normal_()
    memory = torch.randn(1, 7, 8)
    roles = torch.rand(1, 7, 3).softmax(-1)
    expected = layer(memory, roles, 0)
    layer.eval()
    assert torch.equal(layer(memory, roles, 0), expected)
    assert all(not region.merged for region in layer.regions)


def test_second_update_reaches_both_lora_factors_and_all_four_layers():
    heads = RegionalLowRankHeads(channels=8, rank=2)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=0.001)
    memory = torch.randn(1, 7, 8)
    roles = torch.rand(1, 7, 3).softmax(-1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(layer(memory, roles, 0).square().mean() for layer in heads.layers)
        loss.backward()
        optimizer.step()
    assert all(region.lora_A.grad.abs().sum() > 0 and region.lora_B.grad.abs().sum() > 0
               for layer in heads.layers for region in layer.regions)


def test_rejects_invalid_memberships():
    layer = RegionalLowRankLayer(channels=8, rank=2)
    with pytest.raises(ValueError, match="sum to one"):
        layer(torch.ones(1, 7, 8), torch.zeros(1, 7, 3), 0)
