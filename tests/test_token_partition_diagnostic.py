import numpy as np
import pytest
import torch

from modeling.token_partition_diagnostic import partition_tokens


def test_author_partition_preserves_grid_and_is_repeatable():
    # Two contiguous feature populations; neither is declared foreground.
    features = torch.tensor([[1., 0.]] * 8 + [[-1., 0.]] * 8)
    first = partition_tokens(features, 4, 4)
    second = partition_tokens(features, 4, 4)
    assert first["component"].shape == (4, 4)
    assert np.array_equal(first["component"], second["component"])
    assert first["component"].sum() == 8
    assert np.isfinite(first["eigenvector"]).all()


def test_rejects_misaligned_and_invalid_features():
    with pytest.raises(ValueError):
        partition_tokens(torch.ones(15, 2), 4, 4)
    with pytest.raises(ValueError):
        partition_tokens(torch.zeros(16, 2), 4, 4)
    with pytest.raises(ValueError):
        partition_tokens(torch.ones(16, 2, requires_grad=True), 4, 4)
    with pytest.raises(ValueError, match='Degenerate'):
        partition_tokens(torch.ones(16, 2), 4, 4)


def test_prompt_orientation_is_invariant_to_partition_sign():
    from modeling.token_diagnostic_replay import orient_partition
    features = torch.tensor([[1., 0.]] * 8 + [[0., 1.]] * 8)
    partition = np.array([True] * 8 + [False] * 8).reshape(4, 4)
    fg, bg = torch.tensor([1., 0.]), torch.tensor([0., 1.])
    direct, _ = orient_partition(partition, features, fg, bg)
    inverse, _ = orient_partition(~partition, features, fg, bg)
    assert np.array_equal(direct, inverse)
    assert np.array_equal(direct, partition)
    with pytest.raises(ValueError, match='cannot orient'):
        orient_partition(partition, features, fg, fg)


def test_inference_capture_allows_gradient_replay_without_mutation():
    from modeling.token_diagnostic_replay import detached_copy
    with torch.inference_mode():
        original = torch.ones(3, 2)
        copied = detached_copy({'memory': original})['memory']
    copied.requires_grad_(True).square().sum().backward()
    assert torch.equal(copied.grad, torch.full_like(copied, 2))
    assert copied.data_ptr() != original.data_ptr()
    assert torch.equal(original, torch.ones(3, 2))


def test_constant_saliency_and_absent_objects_are_not_reported_as_success():
    from modeling.token_diagnostic_replay import ranking_metrics
    result = ranking_metrics([0, 0, 0, 0], [0, 1, 0, 1])
    assert result['status'] == 'constant_score' and result['auroc'] == .5
    absent = ranking_metrics([0, 1, 2, 3], [0, 0, 0, 0])
    assert absent['auroc'] is None


def test_short_video_coverage_is_resolved_before_inference():
    from modeling.token_diagnostic_replay import diagnostic_target_plan
    with pytest.raises(ValueError, match='exceed'):
        diagnostic_target_plan(0, 15, [1, 2, 4, 8, 16, 32])
    plan = diagnostic_target_plan(0, 15, [1, 2, 4, 8, 16, 32], True)
    assert plan['frames'] == [1, 2, 4, 8]
    assert plan['unavailable_offsets'] == [16, 32]
    with pytest.raises(ValueError):
        diagnostic_target_plan(14, 15, [1], True)
