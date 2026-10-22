"""Native mask semantics, matched nulls, state isolation and paired estimators."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from modeling.region_exclusion import RegionKeyExclusion, exclusion_keys
from scripts.run_region_exclusions import (
    active_bank_digest,
    copy_containers,
    infer_snapshot,
    snapshot_before_frame,
    sequence_statistics,
    report_results,
)


def test_random_exclusion_is_reproducible_and_matched_per_frame():
    labels = torch.tensor(([0] * 8 + [1] * 4 + [2] * 4) * 2 + [-1] * 4)
    units = [
        {"start": 0, "end": 16, "frame_idx": 1},
        {"start": 16, "end": 32, "frame_idx": 2},
    ]
    actual, counts = exclusion_keys(labels, units, 2)
    assert torch.equal(actual, labels == 2)
    for seed in (20261021, 20261022, 20261023):
        random, matched = exclusion_keys(labels, units, 2, seed=seed, context="seq3|4")
        assert matched == counts == [4, 4]
        assert not random[-4:].any()
        assert torch.equal(
            random, exclusion_keys(labels, units, 2, seed=seed, context="seq3|4")[0]
        )
    with pytest.raises(ValueError, match="every key"):
        exclusion_keys(
            torch.zeros(4, dtype=torch.long),
            [{"start": 0, "end": 4, "frame_idx": 1}],
            0,
        )


@pytest.mark.parametrize(
    "device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
)
def test_native_sdpa_mask_excludes_keys_without_changing_layout(device):
    policy = RegionKeyExclusion(SimpleNamespace(training=False), None)
    policy.condition(2, 4, "seq3")
    policy.labels = torch.tensor([0, 1, 2, 0, 1, 2, -1, -1])
    policy.units = []
    policy.selected = policy.labels == 2
    policy.counts = [2]
    policy.bias = None
    q, k, v = (
        torch.randn(1, 1, 4, 8, device=device),
        torch.randn(1, 1, 8, 8, device=device),
        torch.randn(1, 1, 8, 4, device=device),
    )
    output = policy._observe(F.scaled_dot_product_attention, 0, q, k, v)
    keep = (~policy.selected).to(device)
    expected = F.scaled_dot_product_attention(q, k[..., keep, :], v[..., keep, :])
    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-4)
    altered = v.clone()
    altered[..., policy.selected.to(device), :] = 1e5
    assert torch.equal(
        output, policy._observe(F.scaled_dot_product_attention, 1, q, k, altered)
    )
    policy.selected.zero_()
    policy.bias = None
    assert torch.equal(
        policy._observe(F.scaled_dot_product_attention, 2, q, k, v),
        F.scaled_dot_product_attention(q, k, v),
    )


def fake_state():
    outputs = {
        "cond_frame_outputs": {
            0: {"maskmem_features": torch.ones(1, dtype=torch.bfloat16)}
        },
        "non_cond_frame_outputs": {
            1: {"maskmem_features": torch.ones(1)},
            2: {"pred_masks": torch.ones(1), "score": torch.tensor(1.0)},
        },
    }
    return {
        "output_dict": outputs,
        "output_dict_per_obj": {0: copy_containers(outputs)},
        "frames_already_tracked": {0: {}, 1: {}, 2: {}},
        "cached_features": {2: (torch.ones(1), {})},
    }


def test_cached_snapshot_omits_current_frame_and_does_not_write_back():
    original = fake_state()
    snapshot = snapshot_before_frame(original, 2)
    assert 2 not in snapshot["output_dict"]["non_cond_frame_outputs"]
    assert 2 not in snapshot["output_dict_per_obj"][0]["non_cond_frame_outputs"]
    assert 2 not in snapshot["frames_already_tracked"]
    assert snapshot["cached_features"][2][0] is original["cached_features"][2][0]

    class Predictor:
        def run_single_frame_inference(self, **kwargs):
            assert kwargs["run_mem_encoder"] is False
            assert 2 not in kwargs["output_dict"]["non_cond_frame_outputs"]
            kwargs["output_dict"]["non_cond_frame_outputs"][2] = {"wrong": True}
            return {"maskmem_features": None, "maskmem_pos_enc": None}, torch.ones(
                1, 1, 2, 2
            )

        def _get_orig_video_res_output(self, state, low):
            return low, low

    infer_snapshot(Predictor(), snapshot, 2)
    assert 2 not in snapshot["output_dict"]["non_cond_frame_outputs"]
    assert "pred_masks" in original["output_dict"]["non_cond_frame_outputs"][2]


def test_bank_digest_detects_inplace_changes_in_inference_tensors():
    with torch.inference_mode():
        state = fake_state()
        before = active_bank_digest(state, 2, [{"frame_idx": 1}], 16)
        state["output_dict"]["non_cond_frame_outputs"][1]["maskmem_features"].zero_()
        assert before != active_bank_digest(state, 2, [{"frame_idx": 1}], 16)


def test_excess_loss_uses_average_random_not_best(tmp_path):
    def metrics(dice):
        return {
            "dice": dice,
            "iou": dice,
            "boundary_f": dice,
            "false_positive_on_absent": None,
        }

    rows = [
        {
            "region": region,
            "present": True,
            "excluded_tokens": 1,
            "baseline": metrics(0.8),
            "excluded": metrics(0.5),
            "random": [{"scores": metrics(value)} for value in (0.65, 0.7, 0.75)],
        }
        for region in ("background", "foreground_interior", "boundary")
    ]
    stats = sequence_statistics(rows)
    result = stats["boundary"]["present"]["dice"]
    assert result["random"] == pytest.approx(0.7)
    assert result["loss"] == pytest.approx(0.3)
    assert result["excess_loss"] == pytest.approx(0.2)
    assert stats["boundary"]["absent"]["dice"] is None
    report = report_results(
        tmp_path, [{"sequence": "seq3", "status": "completed", "statistics": stats}]
    )
    assert len(report["primary_tests"]) == 3
    assert "Matched random" in (tmp_path / "results.md").read_text()
