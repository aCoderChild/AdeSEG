import numpy as np
import torch

from modeling.region_tokens import (
    category_attention,
    indices_to_masks,
    matched_indices,
    query_regions,
    spatially_matched_indices,
    token_categories,
    persistent_region_rollout,
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


def test_value_substitution_changes_only_selected_spatial_values():
    controller, args = fixture()
    _, memory, units, _, _ = args
    replacement = torch.full((4, 2), 9.0)
    controller.token_value_replacements = {
        0: (torch.tensor([False, True, False, True]), replacement)
    }
    changed = controller.replace_memory_values(memory, units)
    assert changed is not memory
    assert torch.equal(changed[:, 4:], memory[:, 4:])
    assert torch.equal(changed[:, 0], memory[:, 0])
    assert torch.equal(changed[:, 2], memory[:, 2])
    assert torch.equal(changed[:, 1], replacement[1].expand_as(changed[:, 1]))
    assert torch.equal(changed[:, 3], replacement[3].expand_as(changed[:, 3]))
    controller.token_value_replacements = {}
    assert controller.replace_memory_values(memory, units) is memory


def test_value_identity_replacement_is_exact():
    controller, args = fixture()
    _, memory, units, _, _ = args
    controller.token_value_replacements = {
        unit["frame_idx"]: (
            torch.ones(unit["end"] - unit["start"], dtype=torch.bool),
            memory[0, unit["start"] : unit["end"]].clone(),
        )
        for unit in units
    }
    assert torch.equal(controller.replace_memory_values(memory, units), memory)


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


def test_persistent_policy_reads_new_memory_and_native_loop_writes_outputs():
    controller, args = fixture()
    mask = np.zeros((128, 128), bool)
    mask[31:91, 31:91] = True
    state = {"masks": {0: mask}, "written": []}
    reads, diagnostics = [], []

    class Predictor:
        def propagate_in_video(self, state):
            for t in (1, 2):
                units = [
                    {
                        "start": 0,
                        "end": 256,
                        "frame_idx": t - 1,
                        "temporal_distance": 1,
                        "conditioning": False,
                    }
                ]
                selected = controller.token_mask_policy(units)
                assert selected[t - 1].any()
                # Native propagation writes its output before yielding. The
                # region wrapper must neither restore nor replace that output.
                state["masks"][t] = np.roll(state["masks"][t - 1], 1, axis=1)
                state["written"].append(t)
                yield t

    def provider(t):
        reads.append(t)
        return state["masks"][t]

    values = list(
        persistent_region_rollout(
            Predictor(),
            state,
            controller,
            provider,
            "boundary",
            diagnostics=diagnostics,
        )
    )
    assert values == state["written"] == [1, 2]
    assert reads == [0, 1]
    assert all(row["feasible"] for row in diagnostics)
    assert controller.token_mask_policy is None
    assert controller.excluded_token_masks == {}
