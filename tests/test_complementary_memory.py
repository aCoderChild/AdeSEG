"""Controls, soft pooling, object isolation and gradient integrity."""
import copy
import json
import sys

import pytest
import torch
from torch import nn

from modeling.complementary_memory import ComplementaryMemoryReader, NativeComplementaryRead
from modeling.complementary_memory.integration import _memory_alignment_diagnostic


@pytest.fixture
def data():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    return (torch.randn(2, 9, 32), torch.randn(2, 9, 32),
            torch.randn(2, 2, 8, 4, 4), torch.randn(2, 2, 8, 4, 4),
            torch.rand(2, 2, 4, 4))


def module():
    return ComplementaryMemoryReader(memory_dim=8, channels=32, heads=4)


def test_author_loading_does_not_leak_import_paths():
    before = sys.path.copy()
    module()
    assert sys.path == before
    import scripts
    assert "external/Cutie" not in scripts.__file__


def test_exact_identity_then_all_components_receive_gradients(data):
    reader = module()
    optimizer = torch.optim.SGD(reader.parameters(), lr=.01)
    output = reader(*data, "regional")
    assert torch.equal(output, data[0])
    output.square().mean().backward()
    assert reader.output_projection.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = reader(*data, "regional")
    assert not torch.equal(output, data[0])
    output.square().mean().backward()
    for component in (reader.summarizer.feature_pred, reader.summarizer.weights_pred,
                      reader.summarizer.input_proj, reader.reader, reader.position_projection):
        gradients = [p.grad for p in component.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


def test_uses_unchanged_author_soft_pooling(data):
    reader = module()
    _, _, memory, positions, probabilities = data
    actual, _ = reader.summarize(memory, positions, probabilities, "regional")
    raw, _ = reader.summarizer(probabilities, memory)
    expected = (raw[..., :-1] / raw[..., -1:].clamp_min(1e-6)).flatten(1, 2)
    assert torch.equal(actual, expected)


def test_matched_control_ignores_masks_but_regional_uses_them(data):
    reader = module()
    generic = copy.deepcopy(reader)
    assert sum(p.numel() for p in reader.parameters()) == sum(p.numel() for p in generic.parameters())
    _, _, memory, positions, probabilities = data
    a, _ = reader.summarize(memory, positions, probabilities, "regional")
    b, _ = reader.summarize(memory, positions, 1 - probabilities, "regional")
    assert not torch.equal(a, b)
    a, _ = generic.summarize(memory, positions, probabilities, "generic")
    b, _ = generic.summarize(memory, positions, 1 - probabilities, "generic")
    assert torch.equal(a, b)
    uniform, _ = reader.summarize(memory, positions, torch.full_like(probabilities, .5), "regional")
    assert torch.equal(a, uniform)


@pytest.mark.parametrize("probability", [0., 1., .5])
def test_empty_and_uniform_regions_are_finite_without_deleting_summaries(data, probability):
    reader = module()
    data = (*data[:-1], torch.full_like(data[-1], probability))
    with torch.no_grad():
        reader.output_projection.weight.normal_(std=.01)
    output = reader(*data, "regional")
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(p.grad).all() for p in reader.parameters() if p.grad is not None)
    summaries, _ = reader.summarize(*data[2:], "regional")
    assert summaries.shape == (2, 8, 32)


def test_persistent_objects_are_independent_and_permutation_equivariant(data):
    reader = module().eval()
    with torch.no_grad():
        reader.output_projection.weight.normal_(std=.01)
        both = reader(*data, "regional")
        reversed_batch = reader(*(x.flip(0) for x in data), "regional")
        assert torch.allclose(both, reversed_batch.flip(0), atol=1e-6)
        for i in range(2):
            alone = reader(*(x[i:i+1] for x in data), "regional")
            assert torch.allclose(alone, both[i:i+1], atol=1e-6)


def test_invalid_mask_cannot_silently_enter_attention(data):
    bad = data[-1].clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        module()(*data[:-1], bad, "regional")


def test_rejects_unfrozen_original_model():
    model = nn.Linear(2, 2).eval()
    with pytest.raises(RuntimeError, match="frozen"):
        with NativeComplementaryRead(model, module()):
            pass


def test_restore_hooks_after_exception_and_prevent_nested_installation():
    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.memory_attention = nn.Module()
            self.memory_attention.layers = nn.ModuleList(nn.Identity() for _ in range(4))
            self.num_maskmem = 7

        def _prepare_memory_conditioned_features(self):
            pass

    model = Fake().eval()
    original = model._prepare_memory_conditioned_features
    with pytest.raises(ValueError, match="fixture"):
        with NativeComplementaryRead(model, module()):
            with pytest.raises(RuntimeError, match="already"):
                with NativeComplementaryRead(model, module()):
                    pass
            raise ValueError("fixture")
    assert model._prepare_memory_conditioned_features == original
    assert not model.memory_attention._forward_hooks
    assert all(not layer._forward_hooks for layer in model.memory_attention.layers)


def test_memory_mismatch_diagnostic_identifies_span_and_byte_equality():
    context = {"frame": 5, "object_id": 42}
    span = {"frame": 3, "t_pos": 2, "start": 16, "stop": 20}
    output = {"maskmem_features": torch.zeros(1, 1, 2, 2)}
    actual = torch.zeros(4, 1, 1)
    expected = actual.clone()
    expected[0, 0, 0] = 1

    report = json.loads(
        _memory_alignment_diagnostic(context, span, output, actual, expected)
    )

    assert report["current_frame"] == 5
    assert report["object_id"] == 42
    assert report["stored_frame"] == 3
    assert report["memory_slice"] == [16, 20]
    assert report["value_equal"] is False
    assert report["raw_bytes_equal"] is False
    assert report["max_abs_finite_difference"] == 1


def test_memory_mismatch_diagnostic_distinguishes_identical_nan_payloads():
    context = {"frame": 5, "object_id": 42}
    span = {"frame": 3, "t_pos": 2, "start": 0, "stop": 1}
    output = {"maskmem_features": torch.tensor([[[[float("nan")]]]])}
    actual = torch.tensor([[[float("nan")]]])
    expected = actual.clone()

    report = json.loads(
        _memory_alignment_diagnostic(context, span, output, actual, expected)
    )

    assert report["value_equal"] is False
    assert report["raw_bytes_equal"] is True
    assert report["actual_finite_count"] == 0
    assert report["max_abs_finite_difference"] is None
