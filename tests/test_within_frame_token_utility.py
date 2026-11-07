from types import SimpleNamespace

import pytest
import torch

from modeling.sparse_memory import author_sparse_parts, native_positions
from modeling.within_frame_token_utility import (
    ARMS, WithinFrameTokenRead, choose_source, removal_plan,
)


def spans():
    return [{"frame": frame, "t_pos": i, "start": i*4096, "stop": (i+1)*4096}
            for i, frame in enumerate((3, 5, 6, 7))]


def controller():
    read = WithinFrameTokenRead.__new__(WithinFrameTokenRead)
    read.context = {"frame": 8, "object_id": None, "spans": spans()}
    read.selector_mode, read.keep_fraction = "identity", 1.
    read.selection_layers = []
    read.arm, read.source_frame = "native", None
    return read


def test_tiles_partition_exactly_and_unions_are_not_relabelled():
    tiles = torch.stack([removal_plan(f"tile_{i:02d}") for i in range(16)])
    assert (tiles.sum(0) == 1).all()
    assert (tiles.sum(1) == 256).all()
    assert torch.equal(torch.where(tiles[0])[0],
                       (torch.arange(16)[:, None]*64 + torch.arange(16)).flatten())
    assert torch.equal(removal_plan("union_00_15"), tiles[0] | tiles[15])
    assert torch.equal(removal_plan("union_03_12"), tiles[3] | tiles[12])
    assert removal_plan("whole_source").all()
    assert not removal_plan("native").any()


def test_source_plan_deterministic_and_no_protected_or_future_substitution():
    plan = choose_source(spans(), "seq2", 8)
    assert plan == choose_source(spans(), "seq2", 8)
    assert plan["eligible_sources"] == [5, 6]
    assert plan["source_frame"] in [5, 6]
    assert choose_source([spans()[0], spans()[-1]], "seq2", 8)["source_frame"] is None
    broken = spans()
    broken[1]["frame"] = 8
    with pytest.raises(ValueError, match="past-only"):
        choose_source(broken, "seq2", 8)


@pytest.mark.parametrize("arm", ARMS)
def test_only_requested_source_changes_and_author_gather_preserves_every_other_key(arm):
    read = controller()
    read.arm, read.source_frame = arm, 6
    module = SimpleNamespace(training=False)
    pointers = 4
    # Unique token values make accidental source or pointer changes visible.
    k = torch.arange(4*4096+pointers).reshape(1, -1, 1).float()
    v = k + .5
    for layer in range(4):
        read._before_selection(layer, module, (),
                               {"v": v.expand(-1, -1, 64), "num_k_exclude_rope": pointers})
        keep = torch.ones(k.shape[1], dtype=torch.bool)
        keep[8192:12288] = ~removal_plan(arm)
        gather, _ = author_sparse_parts()
        actual_k, actual_v = gather(k, v, module._sparse_mask, 4096, 12288)
        assert torch.equal(actual_k, k[:, keep])
        assert torch.equal(actual_v, v[:, keep])
        assert torch.equal(actual_k[:, -pointers:], k[:, -pointers:])
        assert read.selection_layers[-1]["counts"] == [4096, 4096, int(keep[8192:12288].sum()), 4096]
    assert [r["layer"] for r in read.selection_layers] == [0, 1, 2, 3]


@pytest.mark.parametrize("source", (3, 7, 100))
def test_missing_or_protected_source_fails_closed(source):
    read = controller()
    read.arm, read.source_frame = "tile_00", source
    with pytest.raises(RuntimeError, match="absent or protected"):
        read._before_selection(0, SimpleNamespace(training=False), (),
                               {"v": torch.zeros(1, 16384, 64)})


@pytest.mark.parametrize("arm", ("tile_05", "whole_source", "union_03_12"))
def test_author_rotary_uses_original_cell_coordinates_after_local_removal(arm):
    # Independent reference: rotate the full key grid first, THEN select keys.
    torch.manual_seed(19)
    pos = native_positions()
    q = torch.randn(1, 1, 4096, 8)
    k = torch.randn(1, 1, 4*4096, 8)
    freqs = pos.compute_axial_cis(dim=8, end_x=64, end_y=64)
    q_ref, k_ref = pos.apply_rotary_enc(q, k, freqs_cis=freqs, repeat_freqs_k=True)
    local_keep = ~removal_plan(arm)
    middle = torch.cat((torch.ones(4096, dtype=torch.bool), local_keep))
    all_keep = torch.cat((torch.ones(4096, dtype=torch.bool), middle,
                          torch.ones(4096, dtype=torch.bool)))
    _, rotary = author_sparse_parts()
    q_actual, k_actual = rotary(q, k[:, :, all_keep], freqs_cis=freqs,
        repeat_freqs_k=True, N_k=16384, prune_mask=middle, MTP_id_st=4096, MTP_id_ed=12288)
    assert torch.equal(q_actual, q_ref)
    assert torch.equal(k_actual, k_ref[:, :, all_keep])


def test_read_does_not_compute_or_update_any_selector_cache():
    read = controller()
    read.selection_layers = [{"spatial_keys_after": 6}]
    read.current_pointers = 2
    q, k, v = torch.randn(1, 1, 3, 4), torch.randn(1, 1, 8, 4), torch.randn(1, 1, 8, 4)
    # No masks/anchors/usage properties exist on this fixture.
    assert torch.equal(read._capture_sdpa(q, k, v),
                       torch.nn.functional.scaled_dot_product_attention(q, k, v))
    with pytest.raises(RuntimeError, match="K/V length"):
        read._capture_sdpa(q, k, v[:, :, :-1])
