import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from modeling.contiguous_exclusion import (
    NativeExclusion,
    exclusion_plan,
    rigid_control,
    topology,
)
from scripts.contiguous_exclusion_statistics import summarize_sequence, report
from scripts.run_contiguous_exclusions import snapshot, bank_digest, infer


@pytest.mark.parametrize("kind", ["single", "disconnected", "ring", "background"])
def test_rigid_null_preserves_exact_geometry_and_rng(kind):
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[2:5, 4:8] = True
    if kind == "single":
        mask.zero_()[3, 4] = True
    elif kind == "disconnected":
        mask[8:11, 15:17] = True
    elif kind == "ring":
        mask[3, 5:7] = False
    elif kind == "background":
        mask = ~mask
    original = mask.clone()
    rng = torch.get_rng_state()
    actual, metadata = rigid_control(mask, 12, "fixed")
    repeated, again = rigid_control(mask, 12, "fixed")
    assert torch.equal(mask, original) and torch.equal(rng, torch.get_rng_state())
    assert torch.equal(actual, repeated) and metadata == again
    assert metadata["informative"] and not torch.equal(actual, mask)
    assert topology(actual) == topology(mask)
    if kind == "single":
        assert metadata["candidates"] == 4095


@pytest.mark.parametrize("fill", [False, True])
def test_invariant_masks_cannot_supply_a_null(fill):
    mask = torch.full((64, 64), fill)
    actual, info = rigid_control(mask, 1, "test")
    assert torch.equal(mask, actual)
    assert info["candidates"] == 0 and not info["informative"]


def test_plans_match_each_frame_and_leave_pointers_untouched():
    labels = np.zeros((64, 64), np.uint8)
    labels[4:6, 8:11] = 2
    schemes = {
        0: {name: {"labels": labels} for name in ("a", "b")},
        3: {name: {"labels": np.ones_like(labels)} for name in ("a", "b")},
    }
    units = [{"frame": 0}, {"frame": 3}]
    regional = exclusion_plan(schemes, units, 4, "a", 2, None, "target")
    random = exclusion_plan(schemes, units, 4, "a", 2, 1, "target")
    shared = exclusion_plan(schemes, units, 4, "b", 2, 1, "target")
    assert random["sha256"] == shared["sha256"]
    assert regional["informative"] and random["informative"]
    assert not random["selected"][-4:].any()
    assert random["selected"][:4096].sum() == regional["selected"][:4096].sum() == 6
    assert not random["selected"][4096:].any()
    assert not regional["units"][1]["informative"]
    assert "rotation_quarters" not in regional["units"][0]


def test_native_sdpa_masks_match_explicit_reference_without_qkv_changes():
    control = NativeExclusion(None, "seq2", 0)
    control.active = True
    q, k, v = (torch.randn(1, 1, n, 8) for n in (3, 9, 9))
    before = [(x.clone(), x.stride()) for x in (q, k, v)]
    native = F.scaled_dot_product_attention(q, k, v)
    control.plan = {"selected": torch.zeros(9, dtype=torch.bool), "undefined": False}
    assert torch.equal(control._sdpa(q, k, v), native)
    control.plan["selected"][2:5] = True
    bias = torch.zeros(1, 1, 1, 9)
    bias[..., 2:5] = -torch.inf
    actual = control._sdpa(q, k, v)
    assert torch.equal(actual, F.scaled_dot_product_attention(q, k, v, attn_mask=bias))
    assert torch.allclose(
        actual,
        F.scaled_dot_product_attention(
            q, k[..., [0, 1, 5, 6, 7, 8], :], v[..., [0, 1, 5, 6, 7, 8], :]
        ),
        atol=1e-6,
    )
    assert all(
        torch.equal(x, copy) and x.stride() == stride
        for x, (copy, stride) in zip((q, k, v), before)
    )
    control.plan["undefined"] = True
    with pytest.raises(RuntimeError, match="undefined"):
        control._sdpa(q, k, v)


def state_fixture():
    return {
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {0: {"tensor": torch.ones(2)}},
                "non_cond_frame_outputs": {1: {}, 2: {"tensor": torch.zeros(2)}},
            }
        },
        "frames_tracked_per_obj": {0: {0: {}, 1: {}, 2: {}}},
        "cached_features": {2: (torch.ones(3), {})},
    }


def test_snapshot_filters_current_frame_without_mutating_native_containers():
    state = state_fixture()
    cached = snapshot(state, 2)
    assert set(cached["output_dict_per_obj"][0]["non_cond_frame_outputs"]) == {1}
    assert set(state["output_dict_per_obj"][0]["non_cond_frame_outputs"]) == {1, 2}
    assert 2 not in cached["frames_tracked_per_obj"][0]
    assert cached["cached_features"][2][0] is state["cached_features"][2][0]


def test_bank_hash_detects_target_and_image_mutations_without_writing_state():
    state = state_fixture()
    before = bank_digest(state, 2, [{"frame": 0}], 16)
    assert -1 not in state["output_dict_per_obj"][0]["cond_frame_outputs"]
    state["cached_features"][2][0][0] += 1
    assert bank_digest(state, 2, [{"frame": 0}], 16) != before
    before = bank_digest(state, 2, [{"frame": 0}], 16)
    state["output_dict_per_obj"][0]["non_cond_frame_outputs"][2]["tensor"][0] += 1
    assert bank_digest(state, 2, [{"frame": 0}], 16) != before


def test_counterfactual_uses_author_inference_without_memory_writeback():
    class Predictor:
        def _encode_new_memory(self):
            raise AssertionError("Must not encode")

        def _run_single_frame_inference(self, **kwargs):
            assert kwargs["run_mem_encoder"] is False
            kwargs["output_dict"]["non_cond_frame_outputs"][10] = {}
            return {"maskmem_features": None, "maskmem_pos_enc": None}, torch.ones(
                1, 1, 4, 4
            )

        def _get_orig_video_res_output(self, state, low):
            return None, low

    state = snapshot(state_fixture(), 2)
    assert infer(Predictor(), state, 2).shape == (1, 1, 4, 4)
    assert 10 not in state["output_dict_per_obj"][0]["non_cond_frame_outputs"]


def test_statistics_average_controls_and_exclude_invariant_primary_cases(tmp_path):
    def scores(value):
        return dict.fromkeys(("dice", "iou", "boundary_f"), value)

    row = {
        "definition": "a",
        "region": "boundary",
        "present": True,
        "informative": True,
        "size_bin": "Q1",
        "outcome": "overlap_present",
        "baseline": scores(0.9),
        "excluded": scores(0.4),
        "controls": [{"scores": scores(v)} for v in (0.5, 0.6, 0.7)],
    }
    statistics = summarize_sequence(
        [row, {**row, "informative": False, "excluded": scores(0.1)}]
    )
    primary = next(
        r for r in statistics["rows"] if r["stratum"] == "present_informative"
    )
    assert primary["frames"] == 1
    assert primary["metrics"]["dice"]["excess_loss"] == pytest.approx(0.2)
    protocol = {"definitions": ["a"], "regions": ["boundary"]}
    sequences = [
        {"sequence": f"seq{i}", "status": "completed", "statistics": statistics}
        for i in range(10)
    ]
    result = report(tmp_path, sequences, protocol, "gate")
    assert result["robust_regional_effect"] == {"boundary": False}
    assert not result["primary_comparisons"][0]["passes"]
    json.loads((tmp_path / "exclusion_results.json").read_text())
