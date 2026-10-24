"""Causal, native-mask foreground gating and sampled-frame endpoint checks."""

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from modeling.stale_foreground import StaleForegroundGate, absent_from_score
from scripts.false_alarm_statistics import propagation_summary


class Attention(nn.Module):
    def forward(self, *, q, k, v, num_k_exclude_rope=4):
        return F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]


def setup(score=-1):
    model = nn.Module()
    model.memory_attention = nn.Module()
    model.memory_attention.layers = nn.ModuleList()
    for _ in range(4):
        block = nn.Module()
        block.cross_attn_image = Attention()
        model.memory_attention.layers.append(block)
    model.eval()
    features = torch.arange(64 * 64 * 64, dtype=torch.float32).reshape(1, 64, 64, 64) / 100000
    low = torch.full((1, 1, 256, 256), -2.0)
    low[:, :, 73:184, 73:184] = 2
    output = dict(maskmem_features=features, pred_masks=low,
                  object_score_logits=torch.tensor([[float(score)]]))
    state = {"output_dict_per_obj": {0: {
        "cond_frame_outputs": {0: output}, "non_cond_frame_outputs": {},
    }}}
    v = torch.cat([features.flatten(2).transpose(1, 2), torch.zeros(1, 4, 64)], 1)
    q = torch.ones(1, 2, 64)
    return model, state, output, q, v


def read(model, q, v):
    return [block.cross_attn_image(q=q, k=v, v=v, num_k_exclude_rope=4)
            for block in model.memory_attention.layers]


@pytest.mark.parametrize("score,absent", [(-1, True), (0, True), (1e-20, False)])
def test_native_object_score_boundary(score, absent):
    assert absent_from_score(score) is absent


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), [1, 2], []])
def test_invalid_native_score_fails(score):
    with pytest.raises(ValueError, match="finite"):
        absent_from_score(score)


@pytest.mark.parametrize("enabled,score", [(False, -1), (True, 1)])
def test_disabled_or_present_gate_is_native_bit_exact(enabled, score):
    model, state, _, q, v = setup(score)
    native = read(model, q, v)
    with StaleForegroundGate(model, "seq4", 0, enabled=enabled) as gate:
        gate.capture(state, 0, torch.zeros(1, 1, 1080, 1920))
        actual = read(model, q, v)
        assert gate.mask is None
        assert all(torch.equal(a, b) for a, b in zip(actual, native))
        assert not any(r["gate_active"] for r in gate.trace)


def test_absence_excludes_only_native_stored_foreground_and_keeps_pointers():
    model, state, _, q, v = setup(0)
    native_sdpa = F.scaled_dot_product_attention
    with StaleForegroundGate(model, "seq4", 0, enabled=True) as gate:
        # This passed full-resolution mask deliberately disagrees with storage.
        record = gate.capture(state, 0, torch.full((1, 1, 64, 64), -10.0))
        labels = next(iter(gate.bank.values()))[1].flatten()
        assert set(labels.tolist()) == {0, 1, 2}
        assert record["region_counts"][1] > 0
        actual = read(model, q, v)
        excluded = torch.isneginf(gate.bias)
        assert torch.equal(excluded[:4096], labels == 1)
        assert not excluded[-4:].any()
        assert (gate.bias[:4096][labels != 1] == 0).all()
        expected = native_sdpa(q[:, None], v[:, None], v[:, None], attn_mask=gate.mask)[:, 0]
        assert all(torch.equal(a, expected) for a in actual)
        assert [r["layer"] for r in gate.trace] == [0, 1, 2, 3]
        assert all(r["previous_object_score"] == 0 for r in gate.trace)
    assert F.scaled_dot_product_attention is native_sdpa
    assert not any(b.cross_attn_image._forward_hooks for b in model.memory_attention.layers)


def test_score_is_causal_and_positive_next_frame_restores_native_read():
    model, state, output, q, v = setup(-1)
    with StaleForegroundGate(model, "seq4", 0, enabled=True) as gate:
        gate.capture(state, 0, None)
        read(model, q, v)
        current = dict(output, maskmem_features=output["maskmem_features"] + 1,
                       object_score_logits=torch.tensor([[1.0]]))
        state["output_dict_per_obj"][0]["non_cond_frame_outputs"][1] = current
        record = gate.capture(state, 1, None)
        assert all(r["gate_active"] for r in record["layers"])
        assert record["previous_object_score"] == -1
        assert record["current_object_score"] == 1
        read(model, q, v)
        assert gate.mask is None
        assert not any(r["gate_active"] for r in gate.trace)


def test_ambiguous_memory_labels_and_incomplete_layers_fail():
    model, state, output, q, v = setup()
    with StaleForegroundGate(model, "seq4", 0, enabled=True) as gate:
        gate.capture(state, 0, None)
        state["output_dict_per_obj"][0]["non_cond_frame_outputs"][1] = dict(
            output, pred_masks=-output["pred_masks"])
        with pytest.raises(RuntimeError, match="four"):
            gate.capture(state, 1, None)
        read(model, q, v)
        with pytest.raises(RuntimeError, match="ambiguous"):
            gate.capture(state, 1, None)


def test_exception_restores_native_function_and_nonchronology_fails():
    model, state, _, _, _ = setup()
    original = F.scaled_dot_product_attention
    with pytest.raises(RuntimeError, match="Nonchronological"):
        with StaleForegroundGate(model, "seq4", 0, enabled=True) as gate:
            gate.capture(state, 1, None)
    assert F.scaled_dot_product_attention is original


def row(frame, present, iou=0, fp=False):
    return dict(frame=frame, present=present, absent_fp=fp,
                scores={"iou": iou, "dice": 2 * iou / (1 + iou)})


def test_false_alarms_any_overlap_redetection_and_censoring():
    rows = [row(5, False, fp=True), row(6, True), row(7, True, iou=1e-12),
            row(8, False), row(9, True), row(10, False, fp=True)]
    result = propagation_summary(rows)
    assert result["gt_empty_false_alarm_rate"] == pytest.approx(2 / 3)
    assert result["return_count"] == 2
    assert result["undetected_returns"] == 1
    assert result["returns"][0]["delay_sampled_frames"] == 1
    assert result["returns"][1]["delay_sampled_frames"] is None
    assert result["returns"][1]["censored"]
    assert result["returns"][1]["followup_sampled_frames"] == 1


def test_no_absent_frames_is_undefined_not_zero():
    result = propagation_summary([row(4, True, iou=0.5), row(5, True, iou=1)])
    assert result["gt_empty_false_alarm_rate"] is None
    assert result["return_count"] == 0
    assert result["present_dice"] == pytest.approx((2 / 3 + 1) / 2)


@pytest.mark.parametrize("rows", [[], [row(2, False), row(4, True)],
                                  [row(2, False), row(2, True)]])
def test_nonconsecutive_sampled_targets_fail(rows):
    with pytest.raises(ValueError, match="consecutive"):
        propagation_summary(rows)
