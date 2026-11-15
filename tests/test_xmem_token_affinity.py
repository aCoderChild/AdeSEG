import copy
from argparse import Namespace
import random
import sys
from pathlib import Path

import pytest
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SAM2 = ROOT / "external" / "sam2"
if str(SAM2) not in sys.path:
    sys.path.insert(0, str(SAM2))

from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import Attention, RoPEAttention
from modeling.xmem_token_affinity import (
    XMemTokenAffinityRoPEAttention,
    freeze_sam2_train_xmem_affinity,
    install_xmem_token_affinity,
    xmem_anisotropic_l2,
)
from scripts.verify_xmem_token_affinity import (
    _validate_predicted_memory_writes,
    resolve_gate_settings,
    seed_everything,
    trace_layer0_attention_integration,
    trace_memory_attention_integration,
    record_track_step_write,
    validate_post_update_attention_calls,
)


def _frozen_gate_protocol():
    return {
        "data": {"real_checkpoint_gate_sequences": ["seq2", "seq3"]},
        "engineering_gate": {"settings": {
            "clip_length": 3, "seed": 20261108, "device": "cpu",
        }},
    }


def test_real_checkpoint_gate_defaults_are_protocol_bound():
    args = Namespace(sequences=None, length=None, seed=None, device=None)
    settings = resolve_gate_settings(args, _frozen_gate_protocol())
    assert settings == {
        "sequences": ["seq2", "seq3"], "clip_length": 3,
        "seed": 20261108, "device": "cpu",
    }
    assert args.sequences == settings["sequences"]


@pytest.mark.parametrize("field,value", [
    ("sequences", ["seq3", "seq2"]),
    ("sequences", ["seq2"]),
    ("length", 4),
    ("seed", 7),
    ("device", "mps"),
])
def test_real_checkpoint_gate_rejects_unfrozen_settings(field, value):
    args = Namespace(sequences=None, length=None, seed=None, device=None)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="differs from frozen"):
        resolve_gate_settings(args, _frozen_gate_protocol())


def test_post_update_gate_allows_later_query_changes_but_freezes_memory_geometry():
    native = [{
        "layer": i, "query_sha256": f"native-q{i}", "keys": 194,
        "spatial_keys": 192, "pointer_keys": 2,
        "positioned_key_sha256": f"k{i}", "value_sha256": f"v{i}",
        "pointer_key_sha256": f"pk{i}", "pointer_value_sha256": f"pv{i}",
    } for i in range(4)]
    updated = copy.deepcopy(native)
    for row in updated[1:]:
        row["query_sha256"] = f"changed-q{row['layer']}"
    validate_post_update_attention_calls(native, updated)

    updated[2]["pointer_key_sha256"] = "changed-pointer"
    with pytest.raises(RuntimeError, match="memory/pointer structure"):
        validate_post_update_attention_calls(native, updated)


def test_real_clip_write_gate_accepts_arm_predictions_and_loss_only_ground_truth():
    result = {
        "prompt_index": 0,
        "post_prompt_targets": 2,
        "predicted_memory_writes": 2,
        "ground_truth_used_only_for_loss": True,
        "chunks": [{"writes": [
            {"source": "this_arm_predicted_output", "target_used_for_memory": False},
            {"source": "this_arm_predicted_output", "target_used_for_memory": False},
        ]}],
    }
    assert _validate_predicted_memory_writes(result, 2) is True


def test_real_clip_write_gate_rejects_ground_truth_memory_write():
    result = {
        "prompt_index": 0,
        "post_prompt_targets": 1,
        "predicted_memory_writes": 1,
        "ground_truth_used_only_for_loss": True,
        "chunks": [{"writes": [
            {"source": "ground_truth", "target_used_for_memory": True},
        ]}],
    }
    with pytest.raises(RuntimeError, match="unexpected mask source"):
        _validate_predicted_memory_writes(result, 1)


def test_gate_seed_reproduces_new_adapter_initialization():
    first = _memory_attention()
    seed_everything(20261110)
    install_xmem_token_affinity(first, key_dim=16)
    first_added = {
        name: value.detach().clone()
        for name, value in first.state_dict().items()
        if ".xmem_" in name
    }

    second = _memory_attention()
    seed_everything(20261110)
    install_xmem_token_affinity(second, key_dim=16)
    second_added = {
        name: value.detach().clone()
        for name, value in second.state_dict().items()
        if ".xmem_" in name
    }
    assert first_added.keys() == second_added.keys()
    assert all(torch.equal(first_added[name], second_added[name]) for name in first_added)


def test_gate_seed_resets_python_numpy_and_torch_rngs():
    seed_everything(20261110)
    first = (random.random(), float(np.random.random()), torch.rand(4))
    seed_everything(20261110)
    second = (random.random(), float(np.random.random()), torch.rand(4))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


class _TraceAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.xmem_strength = nn.Parameter(torch.zeros(()))

    def forward(self, q, k, v, num_k_exclude_rope=0):
        return q + self.xmem_strength * (
            k.mean(dim=1, keepdim=True) + v.mean(dim=1, keepdim=True)
        )


class _TraceLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn_image = _TraceAttention()


class _TraceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_attention = nn.Module()
        self.memory_attention.layers = nn.ModuleList([_TraceLayer() for _ in range(4)])

    def track_step(self, frame_idx):
        return {
            name: torch.tensor([float(frame_idx)])
            for name in ("pred_masks", "maskmem_features", "maskmem_pos_enc", "obj_ptr", "object_score_logits")
        }


def test_layer0_integration_trace_records_boundary_and_gradient_without_changing_tensors():
    torch.manual_seed(177)
    q, k, v = torch.randn(1, 4, 8), torch.randn(1, 6, 8), torch.randn(1, 6, 8)
    model = _TraceModel()
    with trace_layer0_attention_integration(model) as rows:
        traced_output = model.memory_attention.layers[0].cross_attn_image(
            q, k, v, num_k_exclude_rope=2
        )
        traced_output.square().sum().backward()

    assert len(rows) == 1
    assert rows[0]["inputs"]["q"]["sha256"]
    assert rows[0]["inputs"]["effective_affinity_q"] == rows[0]["inputs"]["q"]
    assert rows[0]["inputs"]["effective_affinity_memory"] == rows[0]["inputs"]["v"]
    assert rows[0]["inputs"]["pointer_tail_length"] == 2
    assert rows[0]["output"]["sha256"]
    assert rows[0]["incoming_output_gradient"]["sha256"]
    assert rows[0]["incoming_gradient_l1"] > 0

    reference = _TraceModel()
    plain_output = reference.memory_attention.layers[0].cross_attn_image(
        q, k, v, num_k_exclude_rope=2
    )
    plain_output.square().sum().backward()
    assert torch.equal(traced_output, plain_output)
    assert torch.equal(
        model.memory_attention.layers[0].cross_attn_image.xmem_strength.grad,
        reference.memory_attention.layers[0].cross_attn_image.xmem_strength.grad,
    )
    assert not model.memory_attention.layers[0].cross_attn_image._forward_hooks
    assert not model.memory_attention.layers[0].cross_attn_image._forward_pre_hooks


def test_full_memory_attention_trace_covers_each_layer_and_track_step_receipts():
    model = _TraceModel()
    q, k, v = torch.randn(1, 4, 8), torch.randn(1, 6, 8), torch.randn(1, 6, 8)
    with trace_memory_attention_integration(model) as rows:
        output = sum(
            layer.cross_attn_image(q, k, v, num_k_exclude_rope=2).square().sum()
            for layer in model.memory_attention.layers
        )
        output.backward()
    assert [row["layer"] for row in rows] == [0, 1, 2, 3]
    assert all("incoming_output_gradient" in row for row in rows)
    assert all(row["inputs"]["pointer_tail_length"] == 2 for row in rows)

    writes = []
    actual = model.track_step(frame_idx=5)
    record_track_step_write(writes, 5, actual)
    assert len(writes) == 1 and writes[0]["frame_index"] == 5
    for key in ("pred_masks", "maskmem_features", "maskmem_pos_enc", "obj_ptr", "object_score_logits"):
        assert writes[0][key]["shape"] == [1]
        assert writes[0][key]["dtype"] == "torch.float32"
        assert len(writes[0][key]["sha256"]) == 64
        assert torch.equal(actual[key], torch.tensor([5.0]))


def test_freeze_helper_keeps_the_entire_sam2_model_in_eval_mode():
    model = _memory_attention(dropout=0.25).train()
    install_xmem_token_affinity(model, key_dim=16)
    trainable = freeze_sam2_train_xmem_affinity(model)
    assert trainable
    assert not model.training
    assert all(not module.training for module in model.modules())
    assert all(parameter.requires_grad for parameter in trainable)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".xmem_" not in name
    )


def _memory_attention(dropout=0.0):
    cross = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        dropout=dropout,
        kv_in_dim=64,
        rope_k_repeat=True,
        feat_sizes=(8, 8),
    )
    layer = MemoryAttentionLayer(
        activation="relu",
        cross_attention=cross,
        d_model=256,
        dim_feedforward=128,
        dropout=dropout,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention=Attention(256, 1, dropout=dropout),
    )
    return MemoryAttention(
        d_model=256,
        pos_enc_at_input=False,
        layer=layer,
        num_layers=4,
        batch_first=True,
    ).eval()


def _inputs(seed=7):
    torch.manual_seed(seed)
    current = torch.randn(64, 1, 256)
    current_pos = torch.randn_like(current)
    memory = torch.randn(3 * 64 + 2, 1, 64)
    memory_pos = torch.randn_like(memory)
    return current, memory, current_pos, memory_pos


def test_zero_strength_adapter_reproduces_native_four_layer_output_exactly():
    native = _memory_attention()
    adapted = copy.deepcopy(native)
    receipt = install_xmem_token_affinity(adapted, key_dim=16)
    current, memory, current_pos, memory_pos = _inputs()

    expected = native(current, memory, current_pos, memory_pos, num_obj_ptr_tokens=2)
    actual = adapted(current, memory, current_pos, memory_pos, num_obj_ptr_tokens=2)

    assert torch.equal(actual, expected)
    assert receipt["hard_selection"] is False
    assert receipt["frame_selection"] is False
    assert receipt["pointer_affinity"] is False
    assert [entry["adapter_strength"] for entry in receipt["layers"]] == [0.0] * 4
    for layer in adapted.layers:
        assert isinstance(layer.cross_attn_image, XMemTokenAffinityRoPEAttention)
        assert layer.cross_attn_image.xmem_strength.item() == 0.0


def test_adapter_gradient_check_preserves_native_eval_dropout_semantics():
    native = _memory_attention(dropout=0.25).eval()
    adapted = copy.deepcopy(native)
    install_xmem_token_affinity(adapted, key_dim=16)
    trainable = freeze_sam2_train_xmem_affinity(adapted)
    current, memory, current_pos, memory_pos = _inputs(seed=97)

    expected = native(current, memory, current_pos, memory_pos, num_obj_ptr_tokens=2)
    # Match the former gate's wrapper mode: train only the adapted attention
    # modules while leaving the frozen SAM2 layers in eval mode.
    for layer in adapted.layers:
        layer.cross_attn_image.train()
    actual = adapted(current, memory, current_pos, memory_pos, num_obj_ptr_tokens=2)

    assert torch.equal(actual, expected)
    actual.square().mean().backward()
    strengths = [p for p in trainable if p.ndim == 0]
    assert len(strengths) == 4
    assert all(p.grad is not None and torch.isfinite(p.grad) and p.grad.abs() > 0 for p in strengths)
    assert all(p.grad is None for name, p in adapted.named_parameters() if ".xmem_" not in name)


def test_dense_affinity_covers_all_spatial_tokens_and_leaves_pointer_tail_zero():
    attention = XMemTokenAffinityRoPEAttention(
        embedding_dim=256,
        num_heads=1,
        kv_in_dim=64,
        rope_k_repeat=True,
        feat_sizes=(8, 8),
        key_dim=16,
    )
    query = torch.randn(1, 64, 256)
    memory = torch.randn(1, 3 * 64 + 2, 64)
    bias = attention._affinity_module()(query, memory, pointer_tokens=2)

    assert bias.shape == (1, 1, 64, 194)
    assert torch.isfinite(bias).all()
    assert torch.count_nonzero(bias[..., :192]) == 64 * 192
    assert torch.count_nonzero(bias[..., 192:]) == 0


def test_token_treatment_can_assign_different_scores_inside_one_frame():
    torch.manual_seed(43)
    attention = XMemTokenAffinityRoPEAttention(
        embedding_dim=256,
        num_heads=1,
        kv_in_dim=64,
        rope_k_repeat=True,
        feat_sizes=(8, 8),
        key_dim=16,
        granularity="token",
    )
    bias = attention._affinity_module()(
        torch.randn(1, 64, 256), torch.randn(1, 2 * 64 + 1, 64), pointer_tokens=1
    )
    first_frame = bias[0, 0, :, :64]
    assert not torch.equal(first_frame, first_frame[:, :1].expand_as(first_frame))


def test_similarity_matches_direct_xmem_anisotropic_l2_equation():
    torch.manual_seed(41)
    memory_key = torch.randn(2, 5, 7)
    shrinkage = torch.rand(2, 7, 1) + 1.0
    query_key = torch.randn(2, 5, 3)
    query_selection = torch.sigmoid(torch.randn(2, 5, 3))

    actual = xmem_anisotropic_l2(memory_key, shrinkage, query_key, query_selection)
    delta = memory_key.transpose(1, 2)[:, :, None, :] - query_key.transpose(1, 2)[:, None, :, :]
    expected = -(
        delta.square() * query_selection.transpose(1, 2)[:, None, :, :]
    ).sum(-1)
    expected = (expected * shrinkage / (5 ** 0.5)).transpose(1, 2)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_full_query_affinity_matches_concatenated_chunk_scores():
    torch.manual_seed(412)
    attention = XMemTokenAffinityRoPEAttention(
        embedding_dim=256, num_heads=1, kv_in_dim=64,
        rope_k_repeat=True, feat_sizes=(8, 8), key_dim=16,
    )
    query = torch.randn(1, 64, 256)
    memory = torch.randn(1, 3 * 64 + 2, 64)
    projected = attention._affinity_module().project(query, memory, pointer_tokens=2)
    full = attention._affinity_module().score(projected)
    chunked = torch.cat([
        attention._affinity_module().score(projected, start, stop)
        for start, stop in ((0, 19), (19, 43), (43, 64))
    ], dim=1)
    torch.testing.assert_close(full, chunked, rtol=1e-6, atol=1e-6)


def test_learned_dense_affinity_has_finite_gradients_without_dropping_keys():
    attention = XMemTokenAffinityRoPEAttention(
        embedding_dim=256,
        num_heads=1,
        kv_in_dim=64,
        rope_k_repeat=True,
        feat_sizes=(8, 8),
        key_dim=16,
    ).train()
    attention.xmem_strength.data.fill_(0.05)
    query = torch.randn(1, 64, 256)
    memory = torch.randn(1, 3 * 64 + 2, 64)
    pos = torch.randn_like(memory)
    output = attention(
        query,
        memory + pos,
        memory,
        num_k_exclude_rope=2,
        utility_q=query,
        utility_memory=memory,
    )
    output.square().mean().backward()

    adapter_params = [p for name, p in attention.named_parameters() if name.startswith("xmem_")]
    assert adapter_params
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter_params)
    assert all(torch.count_nonzero(p.grad) for p in adapter_params)
    assert output.shape == (1, 64, 256)


def test_integrated_training_updates_only_added_parameters():
    model = _memory_attention()
    install_xmem_token_affinity(model, key_dim=16, granularity="token")
    trainable = freeze_sam2_train_xmem_affinity(model)
    assert trainable
    assert all(".xmem_" in name for name, p in model.named_parameters() if p.requires_grad)
    assert all(not p.requires_grad for name, p in model.named_parameters() if ".xmem_" not in name)

    assert not model.training
    current, memory, current_pos, memory_pos = _inputs(seed=113)
    optimizer = torch.optim.SGD(trainable, lr=0.01)
    # Zero initialization gives exact eval parity; the first update learns the
    # affinity strength, after which the XMem projections receive gradients.
    output = model(current, memory, current_pos, memory_pos, num_obj_ptr_tokens=2)
    output.square().mean().backward()
    assert all(layer.cross_attn_image.xmem_strength.grad is not None for layer in model.layers)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    output = model(current, memory, current_pos, memory_pos, num_obj_ptr_tokens=2)
    output.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
        for p in trainable
    )
    assert all(p.grad is None for name, p in model.named_parameters() if ".xmem_" not in name)


def test_frame_control_broadcasts_full_affinity_within_each_frame():
    attention = XMemTokenAffinityRoPEAttention(
        embedding_dim=256,
        num_heads=1,
        kv_in_dim=64,
        rope_k_repeat=True,
        feat_sizes=(8, 8),
        key_dim=16,
        granularity="frame",
    )
    bias = attention._affinity_module()(
        torch.randn(1, 64, 256), torch.randn(1, 3 * 64 + 2, 64), pointer_tokens=2
    )
    assert bias.shape == (1, 1, 64, 194)
    assert torch.isfinite(bias).all()
    assert torch.count_nonzero(bias[..., :192]) == 64 * 192
    assert torch.count_nonzero(bias[..., 192:]) == 0
    frames = bias[0, 0, :, :192].reshape(64, 3, 64)
    for frame in range(3):
        expected = frames[:, frame, :1].expand_as(frames[:, frame])
        assert torch.equal(frames[:, frame], expected)


def test_native_cross_attention_path_is_unchanged_without_adapter():
    layer = _memory_attention().layers[0]
    before = {name: value.detach().clone() for name, value in layer.cross_attn_image.state_dict().items()}
    current, memory, current_pos, memory_pos = _inputs(seed=29)
    expected = layer._forward_ca(current.transpose(0, 1), memory.transpose(0, 1),
                                 current_pos.transpose(0, 1), memory_pos.transpose(0, 1), 2)
    after = layer._forward_ca(current.transpose(0, 1), memory.transpose(0, 1),
                              current_pos.transpose(0, 1), memory_pos.transpose(0, 1), 2)
    assert torch.equal(expected, after)
    assert all(torch.equal(before[k], v) for k, v in layer.cross_attn_image.state_dict().items())
