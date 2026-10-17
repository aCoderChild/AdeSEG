import pytest
import torch
from torch import nn

from evaluation.utility_retrieval import (
    failure_episodes,
    holm_adjust,
    rank_agreement,
    recovered_fraction,
    temporal_summary,
)
from modeling.utility_memory import UtilityController, UtilityHead
from external.MedSAM2.sam2.modeling.memory_attention import MemoryAttentionLayer
from external.MedSAM2.sam2.modeling.sam.transformer import RoPEAttention


def test_controller_excludes_only_requested_spatial_unit():
    head = UtilityHead(query_dim=4, memory_dim=2, hidden_dim=8)
    controller = UtilityController(head, learn_scale=False)
    controller.enabled = False
    controller.excluded_frame_indices = {3}
    curr = torch.randn(1, 5, 4)
    memory = torch.randn(1, 8, 2)
    units = [
        {"start": 0, "end": 3, "frame_idx": 0, "temporal_distance": 4,
         "conditioning": True, "predicted_iou": torch.tensor([[0.9]]),
         "object_score": torch.tensor([[2.0]])},
        {"start": 3, "end": 6, "frame_idx": 3, "temporal_distance": 1,
         "conditioning": False, "predicted_iou": torch.tensor([[0.5]]),
         "object_score": torch.tensor([[-1.0]])},
    ]
    bias, utilities, features = controller(
        curr, memory, units, num_obj_ptr_tokens=2,
        query_context={"global": curr.mean(1), "foreground": curr[:, :2].mean(1)},
    )
    assert bias.shape == (1, 1, 1, 8)
    assert torch.equal(bias[..., :3], torch.zeros_like(bias[..., :3]))
    assert torch.isneginf(bias[..., 3:6]).all()
    assert torch.equal(bias[..., 6:], torch.zeros_like(bias[..., 6:]))
    assert utilities.shape == (1, 2)
    assert features.shape == (1, 2, head.input_dim)


def test_failure_episodes_are_maximal_and_require_three_frames():
    episodes = failure_episodes([0.8, 0.4, 0.2, 0.49, 0.8, 0.1, 0.2])
    assert [(item.start, item.end) for item in episodes] == [(1, 3)]
    summary = temporal_summary([0.8, 0.4, 0.2, 0.49, 0.8])
    assert summary["episode_count"] == 1
    assert summary["mean_episode_length"] == 3
    assert summary["recovery_rate"] == 1


def test_preregistered_statistics():
    assert recovered_fraction(0.4, 0.1, 0.6) == pytest.approx(0.5)
    agreement = rank_agreement([1, 2, 3], [10, 20, 30])
    assert agreement == {"spearman": 1.0, "kendall": 1.0}
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_rope_attention_zero_bias_is_noop_and_reconstruction_matches():
    torch.manual_seed(7)
    attention = RoPEAttention(
        embedding_dim=8, kv_in_dim=4, num_heads=1, feat_sizes=(2, 2),
        rope_k_repeat=True, dropout=0.0,
    ).eval()
    attention.capture_attention = True
    query = torch.randn(1, 4, 8)
    key = torch.randn(1, 8, 4)
    value = torch.randn(1, 8, 4)
    baseline = attention(query, key, value)
    zero_bias = torch.zeros(1, 1, 1, 8)
    instrumented = attention(query, key, value, attn_mask=zero_bias)
    assert torch.equal(baseline, instrumented)
    assert attention.last_attention.shape == (1, 1, 4, 8)
    assert attention.last_reconstruction_max_abs < 1e-5


def test_value_substitution_keeps_cross_attention_keys_unchanged():
    class IdentityAttention(nn.Module):
        def forward(self, q, k, v, **_):
            return v if v.shape == q.shape else q.new_zeros(q.shape)

    class Recorder(nn.Module):
        def forward(self, q, k, v, **_):
            self.key = k.detach().clone()
            self.value = v.detach().clone()
            return q.new_zeros(q.shape)

    recorder = Recorder()
    layer = MemoryAttentionLayer(
        activation="relu",
        cross_attention=recorder,
        d_model=4,
        dim_feedforward=8,
        dropout=0,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention=IdentityAttention(),
    ).eval()
    target = torch.randn(1, 2, 4)
    memory = torch.randn(1, 3, 4)
    position = torch.randn(1, 3, 4)
    replacement = torch.full_like(memory, 7)
    layer._forward_ca(target, memory, None, position, memory_values=replacement)
    assert torch.equal(recorder.key, memory + position)
    assert torch.equal(recorder.value, replacement)
