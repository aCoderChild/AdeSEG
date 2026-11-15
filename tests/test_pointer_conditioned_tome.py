import pytest
import torch
import torch.nn.functional as F

from modeling.pointer_conditioned_tome import (
    NativePointerConditionedToMe,
    compress_pointer_conditioned_kv,
    pointer_affinity_sequence,
)
from modeling.region_constrained_tome import compress_memory_kv


def _span():
    return [{"frame": 3, "t_pos": 2, "start": 0, "stop": 16}]


def _uniform_roles(batch=1):
    return torch.full((batch, 16, 3), 1 / 3)


def test_same_frame_pointer_affinity_is_aligned_and_objectwise():
    features = torch.randn(2, 4, 4, 4)
    pointer_parts = torch.randn(2, 2, 4)
    outputs = {
        3: {"maskmem_features": features, "obj_ptr": pointer_parts.flatten(1)}
    }
    affinities = pointer_affinity_sequence(_span(), outputs)
    expected = torch.einsum(
        "bnc,brc->bnr",
        F.normalize(features.flatten(2).transpose(1, 2).float(), dim=-1),
        F.normalize(pointer_parts, dim=-1),
    ).mean(-1)
    assert affinities.shape == (2, 16)
    assert torch.equal(affinities, expected)
    assert not torch.equal(affinities[0], affinities[1])


def test_constant_pointer_cue_matches_generic_tome_at_same_budget():
    torch.manual_seed(31)
    keys = torch.randn(1, 2, 18, 8)
    values = torch.randn_like(keys)
    affinities = torch.ones(1, 16)
    pointer = compress_pointer_conditioned_kv(
        keys, values, _span(), affinities, pointers=2, window_size=4,
    )
    plan = {3: {window["window"]: window["merges"]
                for window in pointer[3]["frames"][0]["windows"]}}
    generic = compress_memory_kv(
        keys, values, _uniform_roles(), _span(), mode="global", pointers=2,
        window_size=4, merge_plan=plan,
    )
    assert torch.equal(pointer[0], generic[0])
    assert torch.equal(pointer[1], generic[1])
    assert torch.equal(pointer[2], generic[2])
    assert pointer[3]["spatial_keys_after"] == generic[3]["spatial_keys_after"]


def test_pointer_conditioned_tome_keeps_budget_mass_and_pointer_tail():
    torch.manual_seed(32)
    keys = torch.randn(2, 3, 18, 8)
    values = torch.randn_like(keys)
    affinities = torch.linspace(-1, 1, 16).repeat(2, 1)
    pointer_keys, pointer_values = keys[..., 16:, :].clone(), values[..., 16:, :].clone()
    out_k, out_v, bias, audit = compress_pointer_conditioned_kv(
        keys, values, _span(), affinities, pointers=2, window_size=4,
    )
    assert out_k.shape == out_v.shape == (2, 3, 10, 8)
    assert bias.shape == (1, 1, 1, 10)
    assert torch.equal(out_k[..., -2:, :], pointer_keys)
    assert torch.equal(out_v[..., -2:, :], pointer_values)
    assert audit["spatial_keys_before"] == 16
    assert audit["spatial_keys_after"] == 8
    assert audit["spatial_key_mass_sum"] == 16.0
    assert torch.allclose(bias.exp().sum(), torch.tensor(18.0), atol=1e-5)


def test_pointer_cue_changes_pairing_without_changing_generic_merge_quota():
    torch.manual_seed(33)
    keys = torch.randn(1, 2, 18, 8)
    values = torch.randn_like(keys)
    affinities = torch.randn(1, 16)
    pointer = compress_pointer_conditioned_kv(
        keys, values, _span(), affinities, pointers=2, window_size=4,
    )
    plan = {3: {window["window"]: window["merges"]
                for window in pointer[3]["frames"][0]["windows"]}}
    generic = compress_memory_kv(
        keys, values, _uniform_roles(), _span(), mode="global", pointers=2,
        window_size=4, merge_plan=plan,
    )
    assert pointer[3]["spatial_keys_after"] == generic[3]["spatial_keys_after"] == 8
    assert not torch.equal(pointer[0], generic[0])
    assert not torch.equal(pointer[1], generic[1])


def test_pointer_conditioned_tome_rejects_misaligned_affinities():
    keys = torch.randn(1, 1, 18, 4)
    with pytest.raises(ValueError, match="align with all spatial"):
        compress_pointer_conditioned_kv(
            keys, keys, _span(), torch.zeros(1, 15), pointers=2,
        )


def test_native_wrapper_maps_spans_to_same_frame_memory_and_pointer():
    class Model:
        training = False
        num_maskmem = 2
        memory_temporal_stride_for_eval = 1
        max_cond_frames_in_attn = -1

        def _prepare_memory_conditioned_features(
            self, frame_idx, is_init_cond_frame, current_vision_feats,
            current_vision_pos_embeds, feat_sizes, output_dict, num_frames,
            track_in_reverse=False,
        ):
            return "native result"

    def stored(value):
        return {
            "maskmem_features": torch.full((1, 4, 4, 4), float(value)),
            "obj_ptr": torch.full((1, 8), float(value)),
        }

    bank = {
        "cond_frame_outputs": {0: stored(1)},
        "non_cond_frame_outputs": {1: stored(2)},
    }
    model = Model()
    reader = NativePointerConditionedToMe(model, window_size=4)
    result = reader._prepare(
        frame_idx=2,
        is_init_cond_frame=False,
        current_vision_feats=[torch.zeros(16, 1, 4)],
        current_vision_pos_embeds=[torch.zeros(16, 1, 4)],
        feat_sizes=[(4, 4)],
        output_dict=bank,
        num_frames=3,
        track_in_reverse=False,
    )
    assert result == "native result"
    assert [(span["frame"], span["start"], span["stop"])
            for span in reader.spans] == [(0, 0, 16), (1, 16, 32)]
    assert reader.current_output_dict is None

    reader.current_output_dict = bank
    keys = torch.randn(1, 2, 34, 8)
    values = torch.randn_like(keys)
    compressed = reader._compress_memory(
        keys, values, None, reader.spans, 2, None,
    )
    assert compressed[0].shape[-2] == compressed[1].shape[-2] == 18
    assert compressed[3]["frames"][0]["frame"] == 0
    assert compressed[3]["frames"][1]["frame"] == 1
