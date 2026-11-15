"""Independent numerical and author-memory checks for the current affinity read."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from modeling.prediction_conditioned_affinity import (
    PredictionConditionedRead, PredictionRoleScales, memory_role_sequence,
)
from modeling.native_memory_layout import selected_memory_frames
from modeling.soft_role_affinity import mask_logits_to_soft_roles, soft_mixed_cell_roles


@pytest.mark.parametrize("times, expected", [([2.0, 3.0], 5.0), ([None, 3.0], None)])
def test_cohort_report_preserves_unavailable_runtime(monkeypatch, times, expected):
    from scripts import run_prediction_conditioned_affinity_cv as runner

    monkeypatch.setattr(runner, "SEQUENCES", ["seq2", "seq3"])
    rows = {}
    for name, elapsed in zip(runner.SEQUENCES, times):
        rows[name] = {"present_dice": 0.5, "iou": 0.4, "boundary_f": 0.3,
                      "gt_empty_false_alarm_count": 0}
        if elapsed is not None:
            rows[name]["inference_wall_seconds"] = elapsed
    summary = runner.summarize_arm(rows)
    assert summary["inference_wall_seconds"] == expected
    delta = {"mean_delta": 0.0, "wins": 0, "losses": 0, "ties": 2,
             "ci95": [0.0, 0.0], "wilcoxon": {"p": 1.0}}
    result = {mode: rows for mode in ("native", "global", "regional")}
    result.update(summary={mode: summary for mode in result},
                  regional_vs_native=delta, regional_vs_global=delta,
                  meaningful_signal_screen=False)
    report = runner.report(result)
    assert ("N/A" in report) == (expected is None)


@pytest.fixture
def author_base(monkeypatch):
    names = lambda: [name for name in sys.modules if name == "sam2" or name.startswith("sam2.")]
    previous = {name: sys.modules.pop(name) for name in names()}
    author = str(Path(__file__).resolve().parents[1] / "external/sam2")
    monkeypatch.syspath_prepend(author)
    try:
        from sam2.modeling.sam2_base import SAM2Base
        yield SAM2Base
    finally:
        for name in names():
            sys.modules.pop(name)
        sys.modules.update(previous)


@pytest.mark.parametrize("pointers", [0, 4])
@pytest.mark.parametrize("strength", [0.0, -0.2, 0.2])
def test_chunked_bias_matches_explicit_attention_and_gradients(pointers, strength):
    torch.manual_seed(17)
    q = torch.randn(1, 1, 4096, 8)
    k = torch.randn(1, 1, 9 + pointers, 8)
    v = torch.randn_like(k)
    qr = torch.rand(1, 4096, 3).softmax(-1)
    mr = torch.rand(1, 9, 3).softmax(-1)
    scales = PredictionRoleScales()
    with torch.no_grad():
        scales.raw_strength[0].copy_(torch.tensor([strength, strength / 2, -strength]))
    read = PredictionConditionedRead(None, scales, qr, mr, "regional", chunk_size=257)
    read.sdpa = F.scaled_dot_product_attention
    read.active = (0, pointers)
    actual = read._attention(q, k, v)
    weights = scales.strengths()[0]
    penalty = -(qr[..., None, :] * (1 - mr[:, None]) * weights).sum(-1)
    if pointers:
        penalty = F.pad(penalty, (0, pointers))
    probabilities = (q @ k.transpose(-2, -1) / math.sqrt(8) + penalty[:, None]).softmax(-1)
    expected = probabilities @ v
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    probe = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad((actual * probe).mean(), scales.raw_strength,
                                          retain_graph=True)[0]
    expected_gradient = torch.autograd.grad((expected * probe).mean(), scales.raw_strength)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-8, rtol=1e-4)
    if strength == 0:
        assert torch.equal(actual, F.scaled_dot_product_attention(q, k, v))
    if pointers:
        assert torch.equal(penalty[..., -pointers:], torch.zeros_like(penalty[..., -pointers:]))


@pytest.mark.parametrize("stride", [1, 2, 4])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("frame", [1, 7, 20])
def test_role_spans_match_actual_author_memory_values(stride, reverse, frame, author_base):
    captured = {}

    def attention(**kwargs):
        captured.update(kwargs)
        return kwargs["curr"][0]

    model = SimpleNamespace(
        hidden_dim=256, mem_dim=64, num_maskmem=7, training=False,
        memory_temporal_stride_for_eval=stride, max_cond_frames_in_attn=-1,
        maskmem_tpos_enc=torch.zeros(7, 1, 1, 64),
        use_obj_ptrs_in_encoder=True, max_obj_ptrs_in_encoder=16,
        only_obj_ptrs_in_the_past_for_eval=True, use_signed_tpos_enc_to_obj_ptrs=True,
        add_tpos_enc_to_obj_ptrs=False, memory_attention=attention,
    )

    def stored(index):
        logits = torch.full((1, 1, 256, 256), -8.0)
        logits[..., :4 * (index + 1)] = 8
        return {
            "pred_masks": logits,
            "maskmem_features": torch.full((1, 64, 64, 64), float(index + 1)),
            "maskmem_pos_enc": [torch.zeros(1, 64, 64, 64)],
            "obj_ptr": torch.full((1, 256), float(-100 - index)),
        }

    prompt = 39 if reverse else 0
    bank = {
        "cond_frame_outputs": {prompt: stored(prompt)},
        "non_cond_frame_outputs": {
            i: stored(i) for i in range(40)
            if i != prompt and (i > frame if reverse else i < frame)
        },
    }
    roles, spans = memory_role_sequence(model, frame, bank, 40, reverse)
    assert selected_memory_frames(model, frame, bank, 40, reverse) == spans
    author_base._prepare_memory_conditioned_features(
        model, frame, False, [torch.zeros(4096, 1, 256)],
        [torch.zeros(4096, 1, 256)], [(64, 64)], bank, 40, reverse,
    )
    spatial = captured["memory"].shape[0] - captured["num_obj_ptr_tokens"]
    assert roles.shape[1] == spatial == spans[-1]["stop"]
    for span in spans:
        memory = captured["memory"][span["start"]:span["stop"]]
        assert torch.all(memory == span["frame"] + 1)
        output = bank["cond_frame_outputs"].get(span["frame"],
                     bank["non_cond_frame_outputs"].get(span["frame"]))
        expected = mask_logits_to_soft_roles(output["pred_masks"]).reshape(1, 4096, 3)
        assert torch.equal(roles[:, span["start"]:span["stop"]], expected)
    pointers = captured["memory"][spatial:]
    assert pointers.shape[0] == captured["num_obj_ptr_tokens"] > 0
    assert torch.all(pointers <= -100)


def test_soft_boundary_is_mixedness_and_can_miss_grid_aligned_contours():
    probability = torch.zeros(256, 256)
    probability[:, 128:] = 1
    aligned = soft_mixed_cell_roles(probability)
    assert aligned[..., 2].count_nonzero() == 0
    probability[:, 126:128] = 1
    crossing = soft_mixed_cell_roles(probability)
    assert crossing[:, 31, 2].eq(1).all()
    uncertain = soft_mixed_cell_roles(torch.full((256, 256), 0.5))
    assert uncertain[..., 2].count_nonzero() == 0
    assert uncertain[..., :2].eq(0.5).all()


def test_zero_strength_preserves_upstream_feature_gradients_once():
    torch.manual_seed(23)
    q = torch.randn(1, 1, 4096, 8, requires_grad=True)
    k = torch.randn(1, 1, 9, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    qr = torch.rand(1, 4096, 3).softmax(-1)
    mr = torch.rand(1, 9, 3).softmax(-1)
    scales = PredictionRoleScales()
    read = PredictionConditionedRead(None, scales, qr, mr, "regional")
    read.sdpa = F.scaled_dot_product_attention
    read.active = (0, 0)
    actual = read._attention(q, k, v)
    native = F.scaled_dot_product_attention(q, k, v)
    probe = torch.randn_like(q)
    actual_gradients = torch.autograd.grad((actual * probe).mean(), (q, k, v), retain_graph=True)
    native_gradients = torch.autograd.grad((native * probe).mean(), (q, k, v))
    for actual_gradient, native_gradient in zip(actual_gradients, native_gradients):
        torch.testing.assert_close(actual_gradient, native_gradient, atol=1e-8, rtol=1e-4)


def test_missed_polyp_trains_second_decoder_mask_and_keeps_native_empty_memory(monkeypatch):
    from modeling import prediction_conditioned_training as training

    class Decoder(torch.nn.Module):
        def forward(self, logits):
            return logits, torch.ones(1, 1), None, None

    scales = PredictionRoleScales()
    saved_masks = []
    model = SimpleNamespace(sam_mask_decoder=Decoder())

    class Tracker:
        def __init__(self, model, scales, mode):
            self.reads = []
            self.memory_write_frames = []

        def __enter__(self):
            def step(**kwargs):
                model.sam_mask_decoder(torch.full((1, 1, 256, 256), 5.0))
                model.sam_mask_decoder(scales.raw_strength[0, 0].expand(1, 1, 256, 256))
                self.reads.append({"memory_frames": [], "layers": [{"layer": i} for i in range(4)]})
                output = {
                    "pred_masks": torch.full((1, 1, 256, 256), -1024.0),
                    "object_score_logits": torch.tensor([[-1.0]]),
                    "obj_ptr": torch.zeros(1, 256), "maskmem_pos_enc": [],
                    "maskmem_features": torch.zeros(1, 64, 64, 64),
                }
                if kwargs["run_mem_encoder"]:
                    self.memory_write_frames.append(kwargs["frame_idx"])
                    saved_masks.append(output["pred_masks"].clone())
                return output
            model.track_step = step
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(training, "PredictionConditionedTrackStep", Tracker)
    samples = [{
        "frame": i, "output_dict": {"non_cond_frame_outputs": {}},
        "vision": [], "positions": [], "sizes": [], "num_frames": 3,
        "target": torch.ones(1, 1, 256, 256),
    } for i in (1, 2)]
    loss, records = training.clip_loss(model, scales, samples, "regional")
    loss.backward()
    assert scales.raw_strength.grad[0, 0] < 0
    assert all(row["target_present"] and not row["predicted_present"] for row in records)
    assert len(saved_masks) == 1 and saved_masks[0].eq(-1024).all()
    assert all(row["segmentation_loss"] < 2 for row in records)
