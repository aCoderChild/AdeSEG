import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/sam2"))

from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import Attention
from modeling.position_consistent_tome import (
    PositionConsistentRegionToMe,
    _merge_group,
    build_matching_metric,
    compress_memory_inputs,
    position_indexed_rope_attention,
)


def _roles(labels):
    return F.one_hot(torch.tensor(labels, dtype=torch.long).reshape(1, -1), 3).float()


def _attention():
    return RoPEAttention(
        embedding_dim=32,
        num_heads=4,
        feat_sizes=(4, 4),
        rope_k_repeat=True,
    ).eval()


def _one_head_attention():
    return RoPEAttention(
        embedding_dim=32,
        num_heads=1,
        feat_sizes=(4, 4),
        rope_k_repeat=True,
    ).eval()


def test_joint_key_value_metric_avoids_merging_tokens_with_different_values():
    attention = _one_head_attention()
    with torch.no_grad():
        attention.k_proj.weight.zero_()
        attention.k_proj.bias.zero_()
        attention.v_proj.weight.zero_()
        attention.v_proj.bias.zero_()
        attention.k_proj.weight[:16, :16] = torch.eye(16)
        attention.v_proj.weight[:16, 16:] = torch.eye(16)

    memory = torch.zeros(1, 16, 32)
    position = torch.zeros_like(memory)
    # In ToMe's even/odd split, key-only matching prefers (0, 1): their keys
    # are identical, but their values differ. Joint K/V matching prefers
    # (0, 3), whose keys are close and whose values are identical.
    memory[0, 0, 0] = 1.0
    memory[0, 0, 16] = 1.0
    memory[0, 1, 0] = 1.0
    memory[0, 1, 17] = 1.0
    memory[0, 3, 0] = 0.9
    memory[0, 3, 1] = (1.0 - 0.9**2) ** 0.5
    memory[0, 3, 16] = 1.0
    for index in range(2, 16):
        if index == 3:
            continue
        memory[0, index, index] = 1.0
        memory[0, index, 16 + index] = 1.0

    key_metric = build_matching_metric(attention, memory, position, mode="key")
    joint_metric = build_matching_metric(attention, memory, position, mode="key_value")
    grid = torch.arange(16)
    key_result = _merge_group(
        memory, position, key_metric, grid, 1, position_mode="representative"
    )
    joint_result = _merge_group(
        memory, position, joint_metric, grid, 1, position_mode="representative"
    )

    assert {0, 1} in [set(group) for group in key_result[-1]]
    assert {0, 3} in [set(group) for group in joint_result[-1]]


def _toy_model():
    cross = RoPEAttention(
        embedding_dim=32, num_heads=4, feat_sizes=(4, 4), rope_k_repeat=True
    )
    layer = MemoryAttentionLayer(
        activation="relu",
        cross_attention=cross,
        d_model=32,
        dim_feedforward=64,
        dropout=0.0,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=True,
        self_attention=Attention(32, 4),
    )
    model = _ToySam(MemoryAttention(
        d_model=32,
        pos_enc_at_input=False,
        layer=layer,
        num_layers=4,
        batch_first=True,
    ))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class _ToySam(torch.nn.Module):
    def __init__(self, memory_attention):
        super().__init__()
        self.memory_attention = memory_attention

    def _prepare_memory_conditioned_features(
        self, frame_idx, is_init_cond_frame, current_vision_feats,
        current_vision_pos_embeds, feat_sizes, output_dict, num_frames,
        track_in_reverse=False,
    ):
        return self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=output_dict["memory"],
            memory_pos=output_dict["position"],
            num_obj_ptr_tokens=2,
        )


def test_explicit_positions_reproduce_native_rope_for_unmerged_multiframe_memory():
    torch.manual_seed(12)
    attention = _attention()
    query = torch.randn(1, 16, 32)
    memory = torch.randn(1, 34, 32)
    position = torch.randn_like(memory)
    expected = attention(
        q=query,
        k=memory + position,
        v=memory,
        num_k_exclude_rope=2,
    )
    positions = torch.arange(16).repeat(2)
    actual = position_indexed_rope_attention(
        attention,
        query,
        memory + position,
        memory,
        pointers=2,
        rope_positions=positions,
        attention_bias=torch.zeros(1, 1, 1, 34),
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_merged_positions_preserve_each_frames_temporal_embedding():
    torch.manual_seed(17)
    attention = _attention()
    memory = torch.randn(1, 32, 32)
    spatial_position = torch.randn_like(memory)
    temporal_by_frame = [torch.randn(1, 1, 32), torch.randn(1, 1, 32)]
    position = torch.cat([
        spatial_position[:, :16] + temporal_by_frame[0],
        spatial_position[:, 16:] + temporal_by_frame[1],
    ], dim=1)
    spans = [
        {"frame": 3, "t_pos": 0, "start": 0, "stop": 16},
        {"frame": 4, "t_pos": 6, "start": 16, "stop": 32},
    ]

    merged_memory, merged_position, audit = compress_memory_inputs(
        memory,
        position,
        _roles([0] * 32),
        spans,
        mode="regional",
        pointers=0,
        window_size=2,
        attention=attention,
    )

    output_start = 0
    for span, frame_audit, temporal_embedding in zip(
        spans, audit["frames"], temporal_by_frame, strict=True
    ):
        clusters = [
            cluster
            for window in frame_audit["windows"]
            for cluster in window["clusters"]
        ]
        clusters.sort(key=lambda cluster: cluster["representative_grid_index"])
        output_stop = output_start + frame_audit["output_spatial_keys"]
        actual_position = merged_position[:, output_start:output_stop]
        expected_position_means = []
        recovered_temporal = []
        for cluster in clusters:
            source_indices = torch.tensor(cluster["original_grid_indices"])
            absolute_indices = source_indices + span["start"]
            expected_position_means.append(
                position.index_select(1, absolute_indices).mean(dim=1)[0]
            )
            source_spatial_position = spatial_position.index_select(
                1, absolute_indices
            ).mean(dim=1)[0]
            cluster_position = actual_position[0, len(expected_position_means) - 1]
            recovered_temporal.append(cluster_position - source_spatial_position)

        expected_position_means = torch.stack(expected_position_means).unsqueeze(0)
        recovered_temporal = torch.stack(recovered_temporal)
        assert actual_position.shape[1] == len(clusters)
        assert torch.allclose(
            actual_position, expected_position_means, atol=1e-6, rtol=1e-6
        )
        assert torch.allclose(
            recovered_temporal,
            temporal_embedding[0, 0].expand(len(clusters), -1),
            atol=1e-6,
            rtol=1e-6,
        )
        assert frame_audit["frame"] == span["frame"]
        assert frame_audit["t_pos"] == span["t_pos"]
        output_start = output_stop

    assert output_start == merged_memory.shape[1]


def test_regional_and_generic_merge_match_budgets_and_preserve_pointer_inputs():
    torch.manual_seed(19)
    attention = _attention()
    memory = torch.randn(1, 18, 32)
    position = torch.randn_like(memory)
    roles = _roles([0] * 16)
    spans = [{"frame": 3, "t_pos": 2, "start": 0, "stop": 16}]
    original_pointer_memory = memory[:, 16:].clone()
    original_pointer_position = position[:, 16:].clone()

    regional = compress_memory_inputs(
        memory, position, roles, spans,
        mode="regional", pointers=2, window_size=2, attention=attention,
    )
    region_memory, region_position, region_audit = regional
    plan = {
        3: {
            window["window"]: window["merges"]
            for window in region_audit["frames"][0]["windows"]
        }
    }
    generic_memory, generic_position, generic_audit = compress_memory_inputs(
        memory, position, _roles([1] * 16), spans,
        mode="global", pointers=2, window_size=2,
        merge_plan=plan, attention=attention,
    )

    assert region_memory.shape == generic_memory.shape == (1, 10, 32)
    assert region_position.shape == generic_position.shape == (1, 10, 32)
    assert region_audit["spatial_keys_after"] == generic_audit["spatial_keys_after"] == 8
    assert region_audit["key_mass_sum"] == generic_audit["key_mass_sum"] == 18.0
    assert torch.equal(region_memory[:, -2:], original_pointer_memory)
    assert torch.equal(region_position[:, -2:], original_pointer_position)
    assert torch.equal(generic_memory[:, -2:], original_pointer_memory)
    assert torch.equal(generic_position[:, -2:], original_pointer_position)
    assert region_audit["rope_positions"].shape == (8,)
    assert torch.all((region_audit["rope_positions"] >= 0) & (region_audit["rope_positions"] < 16))
    for window in region_audit["frames"][0]["windows"]:
        top = (window["window"] // 2) * 2
        left = (window["window"] % 2) * 2
        expected_members = sorted([
            top * 4 + left,
            top * 4 + left + 1,
            (top + 1) * 4 + left,
            (top + 1) * 4 + left + 1,
        ])
        clusters = window["clusters"]
        observed_members = sorted(
            index for cluster in clusters for index in cluster["original_grid_indices"]
        )
        assert observed_members == expected_members
        assert all(cluster["mass"] == len(cluster["original_grid_indices"]) for cluster in clusters)
        assert all(cluster["representative_grid_index"] in cluster["original_grid_indices"] for cluster in clusters)

    query = torch.randn(1, 16, 32)
    output = position_indexed_rope_attention(
        attention,
        query,
        region_memory + region_position,
        region_memory,
        pointers=2,
        rope_positions=region_audit["rope_positions"],
        attention_bias=region_audit["attention_bias"],
    )
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_empty_role_groups_are_skipped_without_changing_mass():
    torch.manual_seed(23)
    attention = _attention()
    memory = torch.randn(1, 18, 32)
    position = torch.randn_like(memory)
    labels = [0] * 16
    for index in (1, 5, 9, 13):
        labels[index] = 2
    spans = [{"frame": 1, "t_pos": 1, "start": 0, "stop": 16}]
    _, _, audit = compress_memory_inputs(
        memory, position, _roles(labels), spans,
        mode="regional", pointers=2, window_size=2, attention=attention,
    )
    assert audit["spatial_keys_after"] == 8
    assert audit["spatial_key_mass_sum"] == 16.0


def test_zero_norm_pointer_does_not_invalidate_spatial_matching():
    torch.manual_seed(25)
    attention = _attention()
    with torch.no_grad():
        attention.k_proj.bias.zero_()
    memory = torch.randn(1, 18, 32)
    position = torch.randn_like(memory)
    memory[:, -2:] = 0
    position[:, -2:] = 0
    spans = [{"frame": 1, "t_pos": 1, "start": 0, "stop": 16}]
    _, _, audit = compress_memory_inputs(
        memory, position, _roles([0] * 16), spans,
        mode="regional", pointers=2, window_size=2, attention=attention,
    )
    assert audit["pointer_keys"] == 2
    assert audit["spatial_keys_after"] == 8


def test_zero_merge_preserves_native_spatial_order_and_inputs():
    torch.manual_seed(29)
    attention = _attention()
    memory = torch.randn(1, 18, 32)
    position = torch.randn_like(memory)
    memberships = _roles([0, 1, 2, 0] * 4)
    spans = [{"frame": 3, "t_pos": 2, "start": 0, "stop": 16}]
    plan = {3: {window: 0 for window in range(4)}}

    compressed_memory, compressed_position, audit = compress_memory_inputs(
        memory, position, memberships, spans,
        mode="global", pointers=2, window_size=2,
        merge_plan=plan, attention=attention,
    )

    assert torch.equal(compressed_memory, memory)
    assert torch.equal(compressed_position, position)
    assert torch.equal(audit["rope_positions"], torch.arange(16))
    assert audit["spatial_keys_after"] == 16
    assert audit["spatial_key_mass_sum"] == 16.0


def test_manager_passthrough_is_native_and_restores_patched_methods():
    torch.manual_seed(31)
    native = _toy_model()
    wrapped = _toy_model()
    wrapped.load_state_dict(native.state_dict())
    current = [torch.randn(16, 1, 32)]
    current_pos = [torch.randn(16, 1, 32)]
    memory = torch.randn(18, 1, 32)
    position = torch.randn_like(memory)
    kwargs = dict(
        frame_idx=3,
        is_init_cond_frame=True,
        current_vision_feats=current,
        current_vision_pos_embeds=current_pos,
        feat_sizes=[(4, 4)],
        output_dict={"memory": memory, "position": position},
        num_frames=6,
    )
    expected = native._prepare_memory_conditioned_features(**kwargs)
    original_prepare = wrapped._prepare_memory_conditioned_features
    original_forwards = [layer.cross_attn_image.forward for layer in wrapped.memory_attention.layers]
    original_ca = [layer._forward_ca for layer in wrapped.memory_attention.layers]
    manager = PositionConsistentRegionToMe(wrapped, passthrough=True)
    with manager:
        actual = wrapped._prepare_memory_conditioned_features(**kwargs)
    assert torch.equal(actual, expected)
    assert len(manager.calls) == 4
    assert wrapped._prepare_memory_conditioned_features == original_prepare
    assert [layer.cross_attn_image.forward for layer in wrapped.memory_attention.layers] == original_forwards
    assert [layer._forward_ca for layer in wrapped.memory_attention.layers] == original_ca


def test_manager_applies_compressed_reads_at_all_four_layers(monkeypatch):
    import modeling.position_consistent_tome as module

    torch.manual_seed(37)
    model = _toy_model()
    roles = _roles([0] * 16)
    spans = [{"frame": 3, "t_pos": 2, "start": 0, "stop": 16}]
    monkeypatch.setattr(module, "memory_role_sequence", lambda *args: (roles, spans))
    current = [torch.randn(16, 1, 32)]
    current_pos = [torch.randn(16, 1, 32)]
    memory = torch.randn(18, 1, 32)
    position = torch.randn_like(memory)
    kwargs = dict(
        frame_idx=4,
        is_init_cond_frame=False,
        current_vision_feats=current,
        current_vision_pos_embeds=current_pos,
        feat_sizes=[(4, 4)],
        output_dict={"memory": memory, "position": position},
        num_frames=6,
    )
    manager = PositionConsistentRegionToMe(model, mode="regional", window_size=2)
    with manager:
        output = model._prepare_memory_conditioned_features(**kwargs)
    assert output.shape == current[0].shape
    assert torch.isfinite(output).all()
    assert [call["layer"] for call in manager.calls] == [0, 1, 2, 3]
    assert all(call["spatial_keys_before"] == 16 for call in manager.calls)
    assert all(call["spatial_keys_after"] == 8 for call in manager.calls)
    assert all(call["pointer_keys"] == 2 for call in manager.calls)
    assert all(call["key_mass_sum"] == 18.0 for call in manager.calls)


def test_manager_zero_merge_replays_native_features_at_all_four_layers(monkeypatch):
    import modeling.position_consistent_tome as module

    torch.manual_seed(43)
    native = _toy_model()
    candidate = _toy_model()
    candidate.load_state_dict(native.state_dict())
    roles = _roles([0, 1, 2, 0] * 4)
    spans = [{"frame": 3, "t_pos": 2, "start": 0, "stop": 16}]
    monkeypatch.setattr(module, "memory_role_sequence", lambda *args: (roles, spans))
    current = [torch.randn(16, 1, 32)]
    current_pos = [torch.randn(16, 1, 32)]
    memory = torch.randn(18, 1, 32)
    position = torch.randn_like(memory)
    kwargs = dict(
        frame_idx=4,
        is_init_cond_frame=False,
        current_vision_feats=current,
        current_vision_pos_embeds=current_pos,
        feat_sizes=[(4, 4)],
        output_dict={"memory": memory, "position": position},
        num_frames=6,
    )
    expected = native._prepare_memory_conditioned_features(**kwargs)
    manager = PositionConsistentRegionToMe(
        candidate, mode="no_merge", window_size=2,
        verify_no_merge_parity=True, hash_tensors_target=4,
    )
    with manager:
        actual = candidate._prepare_memory_conditioned_features(**kwargs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert [call["layer"] for call in manager.calls] == [0, 1, 2, 3]
    assert all(call["spatial_keys_after"] == 16 for call in manager.calls)
    assert all(call["pointer_keys"] == 2 for call in manager.calls)
    assert all(call["no_merge_inputs_identity"] for call in manager.calls)
    assert all(call["native_parity_max_abs"] <= 1e-6 for call in manager.calls)
    assert all(
        call["native_output_sha256"] == call["candidate_output_sha256"]
        for call in manager.calls
    )
