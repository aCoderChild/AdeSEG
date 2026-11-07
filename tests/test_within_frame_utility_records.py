import copy
import json
from types import SimpleNamespace

import pytest
from PIL import Image
import numpy as np

from modeling.within_frame_token_utility import ARMS
from scripts.memory_token_io import digest
from scripts.within_frame_utility_records import (
    analyze_target, read_receipt, sample_targets, summarize, write_receipt, write_report,
)


def target(sequence="seq2", frame=6, present=True, native=.8, tile=.79):
    return {"sequence": sequence, "frame": frame, "source_frame": 4,
        "source_quality": {"present": True, "scores": {"dice": .4}},
        "arms": {a: {"frame": frame, "present": present, "absent_fp": not present,
                     "scores": {m: native if a == "native" else tile
                                for m in ("dice", "iou", "boundary_f")}} for a in ARMS}}


def test_utility_sign_union_and_oracle_not_confused_with_a_deployable_selector():
    record = target()
    record["arms"]["tile_00"]["scores"]["dice"] = .83
    record["arms"]["union_00_15"]["scores"]["dice"] = .7
    result = analyze_target(record)
    assert result["tile_utility"]["tile_00"] == pytest.approx(-.03)
    assert result["tile_utility"]["tile_01"] == pytest.approx(.01)
    assert result["mixed_signs"]
    assert result["interactions"]["00_15"] == pytest.approx(.12)
    assert result["best_tile_oracle_arm"] == "tile_00"
    summary = summarize([record])
    assert summary["contrasts"]["best_tile_oracle_dice"]["bootstrap_95_ci"] is None


def test_sequence_units_and_empty_frames_do_not_inflate_dice():
    records = [target(frame=6), target(frame=8), target(frame=9, present=False, native=1., tile=1.),
               target(sequence="seq3", native=.2, tile=.4)]
    summary = summarize(records)
    assert summary["per_sequence"]["seq2"]["native_dice"] == .8
    assert summary["per_sequence"]["seq2"]["empty_targets"] == 1
    assert summary["contrasts"]["uniform_dice"]["mean"] == pytest.approx((-.01+.2)/2)
    assert summary["contrasts"]["uniform_dice"]["leave_one_out_means"]["seq2"] == pytest.approx(.2)
    assert "tile_utility" not in summary["targets"][2]


def test_unmatched_missing_or_duplicated_scientific_records_fail():
    record = target()
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([record, copy.deepcopy(record)])
    record["arms"]["tile_00"]["present"] = False
    with pytest.raises(ValueError, match="denominator"):
        analyze_target(record)
    del record["arms"]["tile_00"]
    with pytest.raises(ValueError, match="Incomplete"):
        analyze_target(record)


def test_sampling_uses_only_frozen_target_ids_and_retains_empty_stratum(tmp_path):
    frames = []
    for i in range(8):
        mask = np.full((2, 2), 1 if i in (1, 3, 5, 7) else 0, np.uint8)
        path = tmp_path/f"{i}.png"
        Image.fromarray(mask).save(path)
        frames.append(SimpleNamespace(mask_path=path, frame_index=i*3))
    plan = sample_targets({"seq2": frames}, {"seq2": {"prompt": 1, "targets": [3, 4, 5, 6, 7]}})
    assert plan["seq2"]["targets"] == [3, 4, 7]
    assert plan["seq2"]["present_targets"] == [3, 7]
    assert plan["seq2"]["empty_targets"] == [4]
    assert plan["seq2"]["source_frame_indices"] == {"3": 9, "4": 12, "7": 21}


def payload(tmp_path):
    (tmp_path/"mask.png").write_bytes(b"fixture")
    return {"context": {"sample_sha256": "native", "source_frame": 5, "arm": "tile_00"},
            "artifacts": {"mask.png": digest(tmp_path/"mask.png")}, "row": {"dice": .8}}


def test_resume_restores_exact_record_and_refuses_modified_context_or_overwrite(tmp_path):
    item = payload(tmp_path)
    path = tmp_path/"receipt.json"
    write_receipt(path, item)
    before = path.read_bytes()
    assert read_receipt(path, item["context"]) == item
    write_receipt(path, item)
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match="context changed"):
        read_receipt(path, {**item["context"], "source_frame": 6})
    with pytest.raises(RuntimeError, match="overwrite"):
        write_receipt(path, {**item, "row": {"dice": .9}})


def test_resume_rejects_receipt_corruption_or_changed_mask(tmp_path):
    item = payload(tmp_path)
    path = tmp_path/"receipt.json"
    write_receipt(path, item)
    (tmp_path/"mask.png").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact changed"):
        read_receipt(path, item["context"])
    raw = json.loads(path.read_text())
    raw["payload"]["row"]["dice"] = .99
    path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match="content changed"):
        read_receipt(path, item["context"])


def test_later_query_preserves_source_and_can_reverse_oracle_benefit():
    first = target()
    first["arms"]["tile_00"]["scores"]["dice"] = .9
    later = target(frame=7, tile=.7)
    later["anchor_target"] = 6
    later["arms"] = {a: later["arms"][a] for a in ARMS[:17]}
    result = summarize([first], [later])
    pair = result["persistence"]["paired_present_targets"][0]
    assert pair["earlier_oracle_selected_tile_later_gain"] == pytest.approx(-.1)
    assert pair["same_sign_fraction"] == 15/16
    later["source_frame"] = 5
    with pytest.raises(ValueError, match="source or timing"):
        summarize([first], [later])


def test_missing_empty_flag_and_boundary_denominator_are_not_silently_filled():
    empty = target(present=False)
    empty["arms"]["tile_00"]["absent_fp"] = None
    with pytest.raises(ValueError, match="false-alarm flag"):
        summarize([empty])
    first, later = target(), target(frame=7)
    later["anchor_target"] = 6
    later["arms"] = {a: later["arms"][a] for a in ARMS[:17]}
    later["arms"]["tile_01"]["present"] = False
    with pytest.raises(ValueError, match="denominator"):
        summarize([first], [later])


def test_report_renders_with_explicit_oracle_and_denominator_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path/"mpl"))
    result = summarize([target(), target(frame=7, present=False)])
    write_report(tmp_path, result)
    report = (tmp_path/"analysis.md").read_text()
    assert "Best tile (oracle)" in report
    assert "Empty targets" in report
    assert "Native boundary F" in report
    with Image.open(tmp_path/"source_quality_vs_utility.png") as figure:
        assert figure.width > 500 and figure.height > 300
