import pytest
import torch

from modeling.lean_memory_position_lookup import MemoryPositionLookup


def test_recovers_permuted_pruned_and_dense_positions_with_temporal_offsets():
    grid = torch.tensor([[1., 10.], [2., 20.], [3., 30.], [4., 40.]])
    offsets = torch.tensor([[0.25, 0.5], [0.75, 1.5]])
    lookup = MemoryPositionLookup(grid, offsets)
    indices = torch.tensor([0, 1, 2, 3, 3, 0, 2, 1])
    pos = torch.cat((grid + offsets[0], grid[indices[4:]] + offsets[1], torch.tensor([[9., 9.]])))[:, None, :]
    assert torch.equal(lookup.decode(pos, 1), indices)


def test_fails_on_unknown_or_ambiguous_spatial_position():
    grid = torch.tensor([[0., 0.], [1., 1.]])
    lookup = MemoryPositionLookup(grid, torch.tensor([[1., 1.]]))
    with pytest.raises(ValueError, match='unambiguously'):
        lookup.decode(torch.tensor([[[1., 1.]]]), 0)
    with pytest.raises(ValueError, match='unambiguously'):
        lookup.decode(torch.tensor([[[9., 9.]]]), 0)
