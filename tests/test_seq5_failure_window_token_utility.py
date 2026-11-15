import pytest
import torch

from modeling.seq5_failure_window_utility import (
    FailureWindowTokenRead,
    TARGETS,
    TILE_NAMES,
    summarize_target,
    target_sources,
)


def _trace(frame=157, ids=(24, 151, 152, 153, 154, 155, 156)):
    return {
        "frame": frame,
        "memory_frames": [
            {"frame": value, "t_pos": 0 if index == 0 else 7 - index}
            for index, value in enumerate(ids)
        ],
    }


def test_targets_and_tile_set_are_frozen():
    assert TARGETS == (157, 184)
    assert TILE_NAMES == tuple(f"tile_{i:02d}" for i in range(16))


def test_actual_intermediate_sources_exclude_prompt_and_latest():
    assert target_sources(_trace()) == [151, 152, 153, 154, 155]


@pytest.mark.parametrize("trace", [
    _trace(ids=(24, 151, 152, 153, 154, 155, 157)),
    _trace(ids=(24, 151, 152, 153, 154, 154, 156)),
    _trace(ids=(24, 151, 152, 153, 154, 155)),
])
def test_source_schedule_rejects_future_duplicate_or_missing_latest(trace):
    with pytest.raises(RuntimeError):
        target_sources(trace)


def test_summary_keeps_oracle_label_and_per_tile_changes():
    native = {"visible": {"scores": {"dice": 0.0}}, "raw": {"scores": {"dice": 0.574}},
              "object_score_logit": -0.883}
    tiles = []
    for index, tile in enumerate(TILE_NAMES):
        dice = 0.1 if index == 2 else 0.0
        tiles.append({"tile": tile, "visible": {"scores": {"dice": dice}},
                      "raw": {"scores": {"dice": 0.574}}, "object_score_logit": -0.8})
    result = summarize_target({"target_frame": 157, "native": native,
                               "sources": [{"source_frame": 155, "tiles": tiles}]})
    source = result["sources"][0]
    assert source["mean_visible_dice_change_vs_native"] == pytest.approx(0.1 / 16)
    assert source["oracle_best_visible_dice_change_vs_native"] == pytest.approx(0.1)
    assert "oracle" in result["warning"].lower()


def test_raw_decoder_capture_selects_the_same_iou_best_candidate():
    reader = object.__new__(FailureWindowTokenRead)
    reader.capture_raw_decoder = True
    reader.raw_decoder_outputs = []
    masks = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    iou = torch.tensor([[0.2, 0.8]])
    score = torch.tensor([[-0.883]])
    reader._capture_decoder(None, None, (masks, iou, torch.zeros(1, 2, 8), score))
    assert len(reader.raw_decoder_outputs) == 1
    result = reader.raw_decoder_outputs[0]
    assert result["selected_mask_index"] == 1
    assert torch.equal(result["lowres_logits"], masks[0, 1])
    assert result["object_score_logit"] == pytest.approx(-0.883)
