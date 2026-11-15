from types import SimpleNamespace

import pytest
import torch

from modeling.native_history_training import block_positions, compact_history_read


def test_block_uses_manifest_positions_without_renumbering_original_ids():
    frames = [SimpleNamespace(frame_index=i) for i in (10, 20, 30, 40, 50, 60)]
    assert block_positions(frames, [30, 40, 50, 60]) == [2, 3, 4, 5]
    for selected in ([10, 30, 40], [20, 20, 30], [20, 30, 99], [20, 30]):
        with pytest.raises(ValueError):
            block_positions(frames, selected)
    with pytest.raises(ValueError):
        block_positions(frames + [frames[0]], [20, 30, 40])


def test_compact_history_preserves_global_time_and_old_pointer_entries():
    def entry(frame):
        return {"maskmem_features": torch.full((1, 64, 2, 2), float(frame)),
                "pred_masks": torch.full((1, 1, 8, 8), float(frame)),
                "maskmem_pos_enc": [torch.ones(1, 64, 2, 2)],
                "obj_ptr": torch.full((1, 256), float(frame)),
                "object_score_logits": torch.tensor([[float(frame)]])}
    sample = {"frame": 32, "num_frames": 300,
              "positions": [torch.ones(3), torch.ones(2), torch.ones(1)],
              "output_dict": {"cond_frame_outputs": {0: entry(0)},
                              "non_cond_frame_outputs": {i: entry(i) for i in range(1, 32)}}}
    packed = compact_history_read(sample, [0, 26, 27, 28, 29, 30, 31])
    assert packed["frame"] == 32 and packed["num_frames"] == 300
    assert set(packed["output_dict"]["non_cond_frame_outputs"]) == set(range(1, 32))
    for frame, value in packed["output_dict"]["non_cond_frame_outputs"].items():
        assert torch.equal(value["obj_ptr"], sample["output_dict"]["non_cond_frame_outputs"][frame]["obj_ptr"])
        assert (value["maskmem_features"] is None) == (frame < 26)
    assert all(value["maskmem_features"] is not None for value in sample["output_dict"]["non_cond_frame_outputs"].values())
    assert [value.numel() for value in packed["positions"]] == [0, 0, 1]
