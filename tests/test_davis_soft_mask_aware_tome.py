import torch

from modeling.davis_soft_mask_aware_tome import DavisObjectAwareToMe
from scripts.run_davis_soft_mask_aware_tome import (
    summarize_official_metrics,
    summarize_local_contrasts,
    verify_observer_gate,
)


class FakePredictor:
    binarize_mask_from_pts_for_mem_enc = False

    def _prepare_memory_conditioned_features(self, frame_idx, output_dict, **kwargs):
        return None

    def track_step(self, frame_idx, output_dict, **kwargs):
        return None

    def _run_memory_encoder(self, inference_state, frame_idx, high_res_masks):
        return None

    def _encode_new_memory(self, pred_masks_high_res, is_mask_from_pts=False):
        return torch.zeros(1, 4, 2, 2), None


def _state(first_logits, second_logits):
    outputs = {
        0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        1: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
    }
    return {
        "output_dict_per_obj": outputs,
        "obj_idx_to_id": {0: 7, 1: 42},
        "temp_output_dict_per_obj": {
            0: {"cond_frame_outputs": {0: {"pred_masks": first_logits[:, :, ::4, ::4],
                                             "maskmem_features": None}},
                "non_cond_frame_outputs": {}},
            1: {"cond_frame_outputs": {0: {"pred_masks": second_logits[:, :, ::4, ::4],
                                             "maskmem_features": None}},
                "non_cond_frame_outputs": {}},
        },
        "device": torch.device("cpu"),
    }


def test_davis_prompt_write_maps_to_exact_persistent_object():
    first = torch.full((1, 1, 8, 8), 2.0)
    second = torch.full((1, 1, 8, 8), -2.0)
    state = _state(first, second)
    assert DavisObjectAwareToMe.prompt_object_id(state, 0, first) == 7
    assert DavisObjectAwareToMe.prompt_object_id(state, 0, second) == 42


def test_davis_object_cache_and_read_routing_do_not_collide_by_frame():
    first = torch.full((1, 1, 8, 8), 2.0)
    second = torch.full((1, 1, 8, 8), -2.0)
    state = _state(first, second)
    model = FakePredictor()
    adapter = DavisObjectAwareToMe(model, state)
    observer = adapter.observer

    first_features = torch.zeros(1, 4, 2, 2)
    second_features = torch.zeros(1, 4, 2, 2)
    adapter.active_object_id = 7
    adapter._capture_memory_write_mask(0, first, first_features)
    adapter.active_object_id = 42
    adapter._capture_memory_write_mask(0, second, second_features)

    assert set(adapter.mask_cache_by_object) == {7, 42}
    assert torch.all(adapter.mask_cache_by_object[7][0][0] > 0.88)
    assert torch.all(adapter.mask_cache_by_object[42][0][0] < 0.12)

    # Exercise the adapter's exact per-read cache routing without invoking SAM2.
    adapter._base_prepare = lambda *args, **kwargs: observer.mask_cache[0][0]
    out7 = state["output_dict_per_obj"][0]
    out42 = state["output_dict_per_obj"][1]
    routed7 = adapter._prepare(frame_idx=1, output_dict=out7)
    routed42 = adapter._prepare(frame_idx=1, output_dict=out42)
    assert torch.equal(routed7, adapter.mask_cache_by_object[7][0][0])
    assert torch.equal(routed42, adapter.mask_cache_by_object[42][0][0])
    assert observer.mask_cache == {}


def _gate_call(object_id, layer, merges=8):
    return {
        "frame": 1,
        "layer": layer,
        "mask_sources": [{
            "object_id": object_id,
            "frame": 0,
            "memory_write_mask_shape": [1024, 1024],
            "memory_grid_shape": [4, 4],
            "occupancy_tokens": 16,
            "memory_mask_binarized": False,
        }],
        "spatial_key_mass_sum": 16.0,
        "key_mass_sum": 17.0,
        "spatial_keys_before": 16,
        "spatial_keys_after": 8,
        "pointer_keys": 1,
        "frames": [{
            "frame": 0,
            "windows": [{
                "window": 0,
                "original_spatial_keys": 16,
                "merges": merges,
                "retained_spatial_keys": 16 - merges,
            }],
        }],
    }


def test_davis_gate_checks_all_objects_layers_and_exact_local_budget():
    observer = type("Observer", (), {})()
    observer.calls = [
        _gate_call(object_id, layer)
        for object_id in (7, 42)
        for layer in range(4)
    ]
    summary = verify_observer_gate(observer, [7, 42], frame_limit=2)
    assert summary["reads"] == 8
    assert summary["checked_windows"] == 8
    assert summary["budget_signature"]["windows"] == 8


def test_davis_gate_rejects_a_window_with_a_different_merge_budget():
    observer = type("Observer", (), {})()
    observer.calls = [
        _gate_call(object_id, layer, merges=7 if object_id == 42 and layer == 2 else 8)
        for object_id in (7, 42)
        for layer in range(4)
    ]
    try:
        verify_observer_gate(observer, [7, 42], frame_limit=2)
    except RuntimeError as error:
        assert "16/8/8 budget" in str(error)
    else:
        raise AssertionError("The gate accepted a mismatched per-window merge budget")


def test_official_davis_metrics_keep_object_rows_and_aggregate_video_unit():
    metrics = {
        "J": {
            "M": [0.8, 0.6, 0.4],
            "R": [0.9, 0.7, 0.5],
            "D": [0.1, 0.2, 0.3],
            "M_per_object": {"video-a_1": 0.8, "video-a_2": 0.6, "video-b_1": 0.4},
        },
        "F": {
            "M": [0.7, 0.5, 0.3],
            "R": [0.8, 0.6, 0.4],
            "D": [0.2, 0.3, 0.4],
            "M_per_object": {"video-a_1": 0.7, "video-a_2": 0.5, "video-b_1": 0.3},
        },
    }
    videos = [
        {"video": "video-a", "object_ids": [1, 2], "frames": ["0", "1", "2", "3"]},
        {"video": "video-b", "object_ids": [1], "frames": ["0", "1", "2"]},
    ]
    summary = summarize_official_metrics(metrics, videos)
    assert len(summary["per_object"]) == 3
    assert abs(summary["global"]["J&F-Mean"] - 0.55) < 1e-12
    assert abs(summary["per_video"]["video-a"]["J&F-Mean"] - 0.65) < 1e-12
    assert abs(summary["per_video"]["video-b"]["J&F-Mean"] - 0.35) < 1e-12
    assert summary["per_video"]["video-a"]["frames_evaluated"] == 2
    assert summary["per_video"]["video-a"]["first_and_last_frames_excluded_by_official_protocol"]


def test_local_davis_contrasts_are_paired_by_video_and_holm_adjusted():
    def condition(offset):
        return {
            "official_evaluation": {
                "per_video": {
                    f"v{i}": {"J&F-Mean": 0.5 + i * 0.01 + offset}
                    for i in range(4)
                }
            }
        }

    result = summarize_local_contrasts({
        "native": condition(0.0),
        "generic_local_tome": condition(-0.01),
        "soft_mask_aware_tome": condition(0.02),
    })
    assert result["soft_mask_aware_tome - native"]["n_videos"] == 4
    assert result["soft_mask_aware_tome - native"]["mean_paired_delta"] > 0
    assert result["soft_mask_aware_tome - generic_local_tome"]["holm_adjusted_p"] >= 0
