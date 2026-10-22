"""Memory-token region geometry, row-major alignment and read-only grouping."""

import numpy as np
import pytest
import torch

from modeling.region_tokens import (
    REGION_NAMES,
    group_memory_vectors,
    reference_token_regions,
)


def test_empty_mask_does_not_create_background_image_border_boundary():
    result = reference_token_regions(np.zeros((80, 120), bool), (8, 12))
    assert (result["labels"] == 0).all()
    assert not result["boundary_coverage"].any()


def test_exactly_three_region_types_and_no_learned_head():
    from modeling import region_tokens

    assert REGION_NAMES == ("background", "foreground_interior", "boundary")
    assert not hasattr(region_tokens, "build_linear_region_probe")
    memory = torch.zeros(1, 4, 1, 3)
    assert tuple(group_memory_vectors(memory, [[0, 1, 2]])) == REGION_NAMES


def test_boundary_adapter_calls_unchanged_author_function(monkeypatch):
    import cv2
    import importlib.util

    original_factory = importlib.util.module_from_spec
    calls = []

    def spy_factory(spec):
        module = original_factory(spec)
        loader = spec.loader
        original_exec = loader.exec_module

        def execute(target):
            original_exec(target)
            author_function = target.mask_to_boundary

            def record(mask, ratio):
                boundary = author_function(mask, ratio)
                calls.append((mask.copy(), ratio, boundary.copy()))
                return boundary

            target.mask_to_boundary = record

        monkeypatch.setattr(loader, "exec_module", execute)
        return module

    monkeypatch.setattr(importlib.util, "module_from_spec", spy_factory)
    mask = np.zeros((80, 120), np.uint8)
    mask[20:60, 30:90] = 1
    result = reference_token_regions(mask, (8, 12))
    assert len(calls) == 2
    assert np.array_equal(calls[1][0], 1 - calls[0][0])
    assert calls[0][1] == calls[1][1]
    pad = result["width_pixels"] + 1
    expected = (calls[0][2] | calls[1][2])[pad:-pad, pad:-pad]
    coverage = cv2.resize(
        expected.astype(np.float32), (12, 8), interpolation=cv2.INTER_AREA
    )
    assert np.array_equal(result["boundary_coverage"], coverage)


def test_region_labels_are_exhaustive_and_boundary_width_is_monotonic():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 1
    results = [
        reference_token_regions(mask, (100, 100), ratio) for ratio in (0.01, 0.02, 0.04)
    ]
    for result in results:
        labels = result["labels"]
        assert set(np.unique(labels)) == {0, 1, 2}
        assert labels[50, 50] == 1 and labels[0, 0] == 0
        assert labels[20, 50] == 2 and labels[19, 50] == 2
    assert all(
        np.all((first["labels"] != 2) | (second["labels"] == 2))
        for first, second in zip(results, results[1:])
    )


def test_mixed_footprints_are_boundary_even_for_tiny_object():
    mask = np.zeros((100, 100), np.uint8)
    mask[44, 44] = 1
    result = reference_token_regions(mask, (10, 10), 0.001)
    assert result["mixed"][4, 4]
    assert result["labels"][4, 4] == 2
    assert result["foreground_coverage"][4, 4] == pytest.approx(0.01)


def test_memory_vectors_reconstruct_exactly_in_row_major_order_without_mutation():
    memory = torch.arange(24).reshape(1, 4, 2, 3).float()
    original = memory.clone()
    labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
    groups = group_memory_vectors(memory, labels)
    assert groups["foreground_interior"]["indices"].tolist() == [1, 4]
    reconstructed = torch.empty(6, 4)
    for group in groups.values():
        reconstructed[group["indices"]] = group["vectors"]
        group["vectors"].zero_()
    assert torch.equal(reconstructed, original.flatten(2).transpose(1, 2)[0])
    assert torch.equal(memory, original)
    with pytest.raises(ValueError, match="invalid"):
        group_memory_vectors(memory, labels + 1)
