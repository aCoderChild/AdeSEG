import torch
from modeling.cutie_memory import CutieMemorySummary, CutieRegionGate


def test_cutie_summary_respects_empty_and_full_masks():
    torch.manual_seed(5)
    model = CutieMemorySummary()
    memory = torch.randn(1, 1, 64, 4, 4, requires_grad=True)
    empty = model(torch.zeros(1, 1, 8, 8), memory)
    full = model(torch.ones(1, 1, 8, 8), memory)
    assert empty.shape == full.shape == (1, 1, 2, 65)
    assert torch.count_nonzero(empty[:, :, 0]) == 0
    assert torch.count_nonzero(full[:, :, 1]) == 0
    assert empty[0, 0, 1, -1] > 0 and full[0, 0, 0, -1] > 0
    (empty.sum() + full.sum()).backward()
    assert memory.grad is not None and torch.isfinite(memory.grad).all()


def test_three_regions_match_author_pool_and_preserve_empty_region():
    torch.manual_seed(6)
    model = CutieMemorySummary()
    memory = torch.randn(1, 1, 64, 4, 4, requires_grad=True)
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    labels[:, 1:3, 1:3] = 1
    result = model.three_regions(labels, memory)
    assert result.shape == (1, 3, 65)
    for region in range(3):
        expected = model((labels == region).float().unsqueeze(1), memory)[:, 0, 0]
        torch.testing.assert_close(result[:, region], expected)
    assert torch.count_nonzero(result[:, 2]) == 0
    result.sum().backward()
    assert memory.grad is not None and torch.isfinite(memory.grad).all()


def test_zero_gate_preserves_attention_and_learned_summary_receives_gradients():
    torch.manual_seed(7)
    gate = CutieRegionGate()
    vectors = torch.randn(12, 64, requires_grad=True)
    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    attention = torch.randn(1, 5, 256)
    context = gate.region_context(vectors, labels, 'regional')
    assert torch.equal(gate(attention, context), attention)
    with torch.no_grad():
        gate.gate.IA.weight.fill_(0.001)
    gate(attention, gate.region_context(vectors, labels, 'regional')).sum().backward()
    assert vectors.grad is not None and torch.isfinite(vectors.grad).all()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in gate.memory_summary.parameters())
