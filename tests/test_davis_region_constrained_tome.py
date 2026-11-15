import pytest

from modeling.davis_region_constrained_tome import (
    DavisObjectAwareRegionToMe,
    matched_plan_from_calls,
    validate_matched_calls,
)


class _Model:
    def _prepare_memory_conditioned_features(self, *args, **kwargs):
        return None


def _state():
    first, second = {}, {}
    return {
        "obj_ids": [3, 19],
        "obj_idx_to_id": {0: 3, 1: 19},
        "output_dict_per_obj": {0: first, 1: second},
    }


def _call(object_id, merges):
    return {
        "frame": 5,
        "object_id": object_id,
        "layer": 0,
        "window_budget_signature": f"sha-{merges}",
        "window_count": 1,
        "frames": [{
            "frame": 2,
            "t_pos": 1,
            "windows": [{
                "window": 0,
                "merges": merges,
                "original_spatial_keys": 16,
                "retained_spatial_keys": 16 - merges,
            }],
        }],
    }


def test_davis_observer_maps_nonconsecutive_persistent_ids_to_distinct_banks():
    state = _state()
    observer = DavisObjectAwareRegionToMe(_Model(), state)
    assert sorted(observer.object_id_by_output_dict.values()) == [3, 19]
    assert observer.object_id_by_output_dict[id(state["output_dict_per_obj"][0])] == 3
    assert observer.object_id_by_output_dict[id(state["output_dict_per_obj"][1])] == 19


def test_davis_observer_rejects_missing_or_duplicate_instance_mapping():
    state = _state()
    state["obj_idx_to_id"] = {0: 3, 1: 3}
    with pytest.raises(RuntimeError, match="Per-object bank map is invalid"):
        DavisObjectAwareRegionToMe(_Model(), state)


def test_matched_plan_is_specific_to_target_and_persistent_object():
    calls = [_call(3, 4), _call(19, 7)]
    calls = [
        {**call, "layer": layer}
        for call in calls
        for layer in range(4)
    ]
    plan = matched_plan_from_calls(calls)
    assert plan[(5, 3)][2][0] == 4
    assert plan[(5, 19)][2][0] == 7
    assert validate_matched_calls(calls, calls)["exact_match"] is True


def test_matched_budget_validation_detects_object_budget_mismatch():
    calls = [{**_call(3, 4), "layer": layer} for layer in range(4)]
    different = [{**_call(3, 5), "layer": layer} for layer in range(4)]
    with pytest.raises(RuntimeError, match="unmatched token budgets"):
        validate_matched_calls(calls, different)


def test_compacted_layer_receipts_preserve_the_exact_reusable_window_plan():
    calls = []
    for layer in range(4):
        item = {**_call(3, 4), "layer": layer}
        if layer:
            item["frames"] = [{**item["frames"][0], "windows": []}]
        calls.append(item)
    plan = matched_plan_from_calls(calls)
    assert plan[(5, 3)] == {2: {0: 4}}
    assert validate_matched_calls(calls, calls)["matched_window_reads"] == 4


def test_matched_budget_validation_rejects_layer_signature_drift():
    regional = [{**_call(3, 4), "layer": layer} for layer in range(4)]
    generic = [{**_call(3, 4), "layer": layer} for layer in range(4)]
    generic[-1]["window_budget_signature"] = "different"
    with pytest.raises(RuntimeError, match="unmatched token budgets"):
        validate_matched_calls(regional, generic)
