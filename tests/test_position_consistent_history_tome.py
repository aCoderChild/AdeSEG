import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/sam2"))

from sam2.modeling.sam.transformer import RoPEAttention
from modeling.position_consistent_history_tome import (
    PositionConsistentLongHistoryToMe,
    build_fixed_budget_merge_plan,
    extend_temporal_position_table,
)
from modeling.position_consistent_tome import compress_memory_inputs
from modeling.wider_history_tome import allocate_history_budget


def _spans(map_tokens, noncond_count):
    spans = [{"frame": 0, "t_pos": 0, "start": 0, "stop": map_tokens}]
    for index in range(noncond_count):
        start = (index + 1) * map_tokens
        spans.append({
            "frame": index + 1,
            "t_pos": index + 1,
            "start": start,
            "stop": start + map_tokens,
        })
    return spans


def test_fixed_budget_plan_conserves_exact_per_frame_and_local_quotas():
    for allocation in ("uniform", "recent_dense"):
        for count in range(1, 13):
            spans = _spans(64, count)
            quotas = allocate_history_budget(
                spans,
                map_tokens=64,
                native_noncond_maps=6,
                max_noncond_maps=12,
                allocation=allocation,
            )
            plan = build_fixed_budget_merge_plan(
                spans,
                map_tokens=64,
                native_noncond_maps=6,
                max_noncond_maps=12,
                window_size=4,
                allocation=allocation,
            )
            assert set(plan) == {span["frame"] for span in spans}
            for index, span in enumerate(spans):
                source = span["stop"] - span["start"]
                assert sum(plan[span["frame"]].values()) == source - quotas[index]
                assert len(plan[span["frame"]]) == 4
                assert max(plan[span["frame"]].values()) <= 8
            if count > 6:
                assert sum(quotas[index] for index in range(1, len(spans))) == 6 * 64


def test_position_consistent_compression_obeys_budget_and_preserves_pointer_tail():
    torch.manual_seed(271)
    map_tokens = 64
    spans = _spans(map_tokens, 7)
    spatial_tokens = map_tokens * len(spans)
    pointers = 2
    memory = torch.randn(1, spatial_tokens + pointers, 32)
    position = torch.randn_like(memory)
    pointer_memory = memory[:, -pointers:].clone()
    pointer_position = position[:, -pointers:].clone()
    roles = F.one_hot(torch.zeros(1, spatial_tokens, dtype=torch.long), 3).float()
    plan = build_fixed_budget_merge_plan(
        spans,
        map_tokens=map_tokens,
        native_noncond_maps=6,
        max_noncond_maps=7,
        window_size=4,
        allocation="uniform",
    )
    quotas = allocate_history_budget(
        spans,
        map_tokens=map_tokens,
        native_noncond_maps=6,
        max_noncond_maps=7,
        allocation="uniform",
    )
    attention = RoPEAttention(
        embedding_dim=32,
        num_heads=4,
        feat_sizes=(8, 8),
        rope_k_repeat=True,
    ).eval()

    out_memory, out_position, audit = compress_memory_inputs(
        memory,
        position,
        roles,
        spans,
        mode="global",
        pointers=pointers,
        window_size=4,
        merge_plan=plan,
        attention=attention,
        position_mode="representative",
        matching_mode="key_value",
    )

    expected_spatial_tokens = map_tokens + 6 * map_tokens
    assert out_memory.shape == out_position.shape == (1, expected_spatial_tokens + pointers, 32)
    assert torch.equal(out_memory[:, -pointers:], pointer_memory)
    assert torch.equal(out_position[:, -pointers:], pointer_position)
    assert audit["spatial_keys_after"] == expected_spatial_tokens
    assert audit["matching_mode"] == "key_value"
    assert audit["key_mass_sum"] == spatial_tokens + pointers
    expected_position_rows = []
    for index, span in enumerate(spans):
        frame = audit["frames"][index]
        assert frame["frame"] == span["frame"]
        assert frame["output_spatial_keys"] == quotas[index]
        assert sum(window["merges"] for window in frame["windows"]) == frame["merged_keys"]
        representatives = sorted(
            cluster["representative_grid_index"]
            for window in frame["windows"]
            for cluster in window["clusters"]
        )
        expected_position_rows.extend(
            position[0, int(span["start"]) + representative]
            for representative in representatives
        )
    expected_positions = torch.stack(expected_position_rows).unsqueeze(0)
    assert torch.equal(out_position[:, :expected_spatial_tokens], expected_positions)


def test_temporal_extension_preserves_native_rows_and_reuses_only_oldest_row():
    original = torch.arange(7 * 2, dtype=torch.float32).reshape(7, 1, 1, 2)
    extended = extend_temporal_position_table(original, max_noncond_maps=10)
    assert extended.shape == (11, 1, 1, 2)
    assert torch.equal(extended[:6], original[:6])
    assert torch.equal(extended[6:10], original[5:6].expand(4, 1, 1, 2))
    assert torch.equal(extended[10:11], original[6:7])
    for age in range(1, 7):
        native_t_pos = 7 - age
        extended_t_pos = 11 - age
        native_index = 7 - native_t_pos - 1
        extended_index = 11 - extended_t_pos - 1
        assert torch.equal(extended[extended_index], original[native_index])
    for age in range(7, 11):
        extended_t_pos = 11 - age
        extended_index = 11 - extended_t_pos - 1
        assert torch.equal(extended[extended_index], original[5])


class _ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn_image = RoPEAttention(
            embedding_dim=32,
            num_heads=4,
            feat_sizes=(8, 8),
            rope_k_repeat=True,
        )
        self.pos_enc_at_cross_attn_keys = True

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        return tgt


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_attention = torch.nn.Module()
        self.memory_attention.layers = torch.nn.ModuleList([_ToyLayer() for _ in range(4)])
        self.num_maskmem = 7
        self.maskmem_tpos_enc = torch.nn.Parameter(torch.randn(7, 1, 1, 32))

    def _prepare_memory_conditioned_features(self, frame_idx, is_init_cond_frame):
        return frame_idx, is_init_cond_frame


def test_long_history_context_restores_native_temporal_state():
    model = _ToyModel().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    original_tpos = model.maskmem_tpos_enc.detach().clone()
    original_num_maskmem = model.num_maskmem

    with PositionConsistentLongHistoryToMe(model, max_noncond_maps=10):
        assert model.num_maskmem == 11
        assert model.maskmem_tpos_enc.shape[0] == 11
        assert torch.equal(model.maskmem_tpos_enc.detach()[:6], original_tpos[:6])
        assert torch.equal(
            model.maskmem_tpos_enc.detach()[6:10],
            original_tpos[5:6].expand(4, 1, 1, 32),
        )
        assert torch.equal(model.maskmem_tpos_enc.detach()[10:11], original_tpos[6:7])

    assert model.num_maskmem == original_num_maskmem
    assert torch.equal(model.maskmem_tpos_enc.detach(), original_tpos)
    mean_control = PositionConsistentLongHistoryToMe(
        model, max_noncond_maps=10, position_mode="mean"
    )
    assert mean_control.position_mode == "mean"
