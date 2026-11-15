"""Scientific invariants of the training-only contrastive integration."""
import types

import pytest
import torch
import torch.nn.functional as F

from modeling.complementary_memory import ComplementaryMemoryReader
from modeling.complementary_memory.contrastive import (
    ForegroundContextContrast, author_supcon, region_prototypes,
)


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(101)
    reader = ComplementaryMemoryReader(memory_dim=8, channels=32, heads=4)
    data = (torch.randn(1, 16, 32), torch.randn(1, 16, 32),
            torch.randn(1, 2, 8, 4, 4), torch.randn(1, 2, 8, 4, 4),
            torch.rand(1, 2, 4, 4))
    target = torch.zeros(1, 1, 16, 16)
    target[..., :8, :] = 1
    observer = types.SimpleNamespace(calls=[{
        "frame": 2, "memory_frames": [{"frame": 0}, {"frame": 1}],
    }])
    return reader, data, target, observer


def test_upstream_loss_rewards_correct_feature_separation():
    loss = author_supcon()()
    features = F.normalize(torch.tensor([[1., .01], [1., -.01], [-1., .01], [-1., -.01]]), dim=-1)
    correct = loss(features[:, None], torch.tensor([1, 1, 0, 0]))
    crossed = loss(features[:, None], torch.tensor([1, 0, 1, 0]))
    assert torch.isfinite(correct) and correct < crossed


def test_fractional_boundary_occupancy_and_missing_class():
    values = torch.tensor([[2., 0.], [0., 4.]])
    prototypes, labels = region_prototypes(values, torch.tensor([1., .5]))
    assert labels == [1, 0]
    assert torch.allclose(prototypes[0], torch.tensor([4/3, 4/3]))
    assert torch.equal(prototypes[1], values[1])
    empty, labels = region_prototypes(values, torch.zeros(2))
    assert labels == [0] and torch.equal(empty[0], values.mean(0))


def test_gt_changes_loss_but_never_forward_or_parameters():
    reader, data, target, observer = fixture()
    before = {k: v.clone() for k, v in reader.state_dict().items()}
    with torch.no_grad():
        reader.output_projection.weight.normal_(std=.01)
    before = {k: v.clone() for k, v in reader.state_dict().items()}
    outputs, losses = [], []
    for query_target in (target, 1 - target):
        with ForegroundContextContrast(reader, observer, {0: target, 1: target}) as auxiliary:
            outputs.append(reader(*data, "regional"))
            losses.append(auxiliary({"frame": 2, "target": query_target}))
    assert torch.equal(outputs[0], outputs[1])
    assert not torch.equal(losses[0], losses[1])
    assert all(torch.equal(before[k], v) for k, v in reader.state_dict().items())
    assert not reader.summarizer.feature_pred._forward_hooks
    assert not reader.reader._forward_pre_hooks


def test_contrast_reaches_actual_query_key_and_memory_features():
    reader, data, target, observer = fixture()
    with ForegroundContextContrast(reader, observer, {0: target, 1: target}) as auxiliary:
        result = reader(*data, "regional")
        loss = auxiliary({"frame": 2, "target": target})
    assert torch.equal(result, data[0])  # zero-residual identity
    loss.backward()
    grad = reader.reader.cross_attn.in_proj_weight.grad
    assert grad[:32].abs().sum() > 0 and grad[32:64].abs().sum() > 0
    assert grad[64:].abs().sum() == 0  # contrast operates on Q/K, not V
    assert reader.summarizer.input_proj.weight.grad.abs().sum() > 0
    assert reader.output_projection.weight.grad is None


def test_zero_weight_control_has_zero_auxiliary_gradient():
    reader, data, target, observer = fixture()
    with ForegroundContextContrast(reader, observer, {0: target, 1: target}, weight=0) as auxiliary:
        reader(*data, "regional")
        loss = auxiliary({"frame": 2, "target": target})
    loss.backward()
    assert float(loss) == 0
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in reader.parameters())


def test_future_frame_and_missing_receipts_fail_closed():
    reader, data, target, observer = fixture()
    observer.calls[0]["memory_frames"][1]["frame"] = 2
    with ForegroundContextContrast(reader, observer, {0: target, 1: target}) as auxiliary:
        reader(*data, "regional")
        with pytest.raises(RuntimeError, match="causal"):
            auxiliary({"frame": 2, "target": target})
    assert not reader.reader._forward_pre_hooks


def test_all_background_supervision_is_finite_zero():
    reader, data, target, observer = fixture()
    target.zero_()
    with ForegroundContextContrast(reader, observer, {0: target, 1: target}) as auxiliary:
        reader(*data, "regional")
        loss = auxiliary({"frame": 2, "target": target})
    assert torch.isfinite(loss) and float(loss) == 0


@pytest.mark.parametrize("gain,harm,expected", [(0., 0., False), (.03, 0., True), (.03, .10, False)])
def test_report_applies_frozen_effect_and_harm_rules(tmp_path, gain, harm, expected):
    from scripts.run_foreground_context_contrast import base, report
    summaries = {}
    counts = [33, 46, 62, 55, 35, 47, 50, 44]  # exactly 372 synthetic targets
    for arm, score, fa in (("native", .7, .1), ("no_contrast", .7, .1),
                           ("contrastive", .7 + gain, .1 + harm)):
        summaries[arm] = {name: {
            "target_frames": count, "present_dice": score,
            "present_iou": .6, "present_boundary_f": .5,
            "gt_empty_false_alarm_count": round(fa * 10), "gt_empty_frames": 10,
            "gt_empty_false_alarm_rate": fa, "elapsed_seconds": 1.,
        } for name, count in zip(base.TEST_SEQUENCES, counts)}
    result = report(summaries, tmp_path)
    assert result["pilot_screen_passed"] is expected
    assert result["means"]["native"]["present_dice"] == pytest.approx(.7)
