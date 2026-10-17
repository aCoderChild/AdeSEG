import numpy as np
import torch

from modeling.region_tokens import (
    category_attention,
    indices_to_masks,
    matched_indices,
    query_regions,
    spatially_matched_indices,
    token_categories,
)
from modeling.utility_memory import UtilityController, UtilityHead


def fixture():
    controller = UtilityController(UtilityHead(4, 2, 8), learn_scale=False)
    controller.enabled = False
    curr, memory = torch.randn(1, 4, 4), torch.randn(1, 10, 2)
    units = [
        {
            "start": 0,
            "end": 4,
            "frame_idx": 0,
            "conditioning": True,
            "temporal_distance": 2,
        },
        {
            "start": 4,
            "end": 8,
            "frame_idx": 1,
            "conditioning": False,
            "temporal_distance": 1,
        },
    ]
    context = {"global": curr.mean(1), "foreground": curr.mean(1)}
    return controller, (curr, memory, units, 2, context)


def test_empty_and_full_token_masks_match_original_paths():
    controller, args = fixture()
    assert controller(*args)[0] is None
    controller.excluded_token_masks = {0: torch.zeros(4, dtype=torch.bool)}
    assert controller(*args)[0] is None
    controller.excluded_token_masks = {0: torch.ones(4, dtype=torch.bool)}
    token_bias = controller(*args)[0]
    controller.excluded_token_masks = {}
    controller.excluded_frame_indices = {0}
    assert torch.equal(token_bias, controller(*args)[0])
    assert (token_bias[..., 8:] == 0).all()


def test_pointer_reference_does_not_mask_spatial_keys():
    controller, args = fixture()
    controller.exclude_pointers = True
    bias = controller(*args)[0]
    assert torch.isneginf(bias[..., -2:]).all()
    assert (bias[..., :-2] == 0).all()


def test_taxonomy_partition_artifact_and_no_budget_reduction():
    mask = np.zeros((256, 256), bool)
    mask[79:171, 79:171] = True
    result = token_categories(mask, (32, 32), image=np.full((256, 256, 3), 255))
    labels = result["categories"]
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert result["artifact"].all()
    assert matched_indices(labels, 1, labels.size, 0) is None
    a = matched_indices(labels, 0, 3, 2)
    assert len(np.unique(a)) == 3
    assert np.array_equal(a, matched_indices(labels, 0, 3, 2))
    assert len(spatially_matched_indices(labels, a, 2)) == 3


def test_global_selected_indices_map_without_pointer_changes():
    _, args = fixture()
    masks = indices_to_masks(args[2], np.array([1, 4, 7]))
    assert masks[0].tolist() == [False, True, False, False]
    assert masks[1].tolist() == [True, False, False, True]


def test_query_partition_and_attention_mass_conservation():
    foreground = torch.zeros(1, 16, dtype=torch.bool)
    foreground[:, 5:7] = True
    regions = query_regions(foreground, 4, 4)
    assert torch.stack(list(regions.values())).int().sum(0).eq(1).all()
    units = [{"start": 0, "end": 4, "frame_idx": 0}]
    output = category_attention(
        [torch.full((1, 1, 16, 6), 1 / 6)], units, {0: np.array([0, 1, 2, 3])}, regions
    )
    for values in output[0].values():
        if values:
            assert abs(sum(values.values()) - 1) < 1e-6
