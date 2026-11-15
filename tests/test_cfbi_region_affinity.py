import torch

from modeling.cfbi_region_affinity import (
    RegionalAffinityHead,
    RegionalAffinityHeads,
    author_global_matching,
    regional_matching_evidence,
)


def test_pinned_cfbi_matcher_returns_one_map_per_region():
    query = torch.randn(1, 4096, 256)
    memory = torch.randn(4096, 1, 256)
    labels = torch.arange(4096).remainder(3)[None]
    evidence, present = regional_matching_evidence(query, memory, labels, chunks=8)
    assert evidence.shape == (1, 3, 16, 16)
    assert torch.isfinite(evidence).all()
    assert present.tolist() == [True, True, True]
    assert torch.all(evidence.std(dim=(0, 2, 3), unbiased=False) > 1e-6)
    assert evidence.max() < 1.0


def test_absent_region_has_neutral_zero_evidence():
    query = torch.randn(1, 4096, 256)
    memory = torch.randn(4096, 1, 256)
    labels = torch.ones((1, 4096), dtype=torch.long)
    evidence, present = regional_matching_evidence(query, memory, labels, chunks=8)
    assert present.tolist() == [False, True, False]
    assert torch.count_nonzero(evidence[:, [0, 2]]) == 0


def test_zero_affinity_projection_is_exact_native_and_trainable():
    head = RegionalAffinityHead()
    native = torch.randn(1, 256, 64, 64)
    evidence = torch.randn(1, 3, 16, 16)
    output = head(native, evidence)
    assert torch.equal(output, native)
    output.square().mean().backward()
    assert head.projection.weight.grad is not None
    assert torch.isfinite(head.projection.weight.grad).all()
    assert head.projection.weight.grad.abs().sum() > 0


def test_loaded_function_is_author_cfbi_entrypoint():
    assert author_global_matching().__name__ == "global_matching"


def test_region_heads_use_four_attention_layers():
    heads = RegionalAffinityHeads(channels=256)
    assert len(heads.layers) == 4
