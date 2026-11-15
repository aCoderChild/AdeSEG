import json

import pytest

from scripts import run_lean_polypgen_full as study


def test_paired_comparison_checks_content_and_empty_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "SEQUENCES", ["seq3"])
    for arm, dice, absent in (("lean", 0.8, True), ("native", 0.6, False)):
        folder = tmp_path / arm / "seq3"
        folder.mkdir(parents=True)
        (folder / "inputs.json").write_text(json.dumps([
            {"frame": 0, "image_sha256": "image", "mask_sha256": "mask"}
        ]))
        rows = [
            {"source_frame_index": 1, "present": True, "scores": {"dice": dice}, "absent_fp": None},
            {"source_frame_index": 2, "present": False, "scores": {"dice": 0.0}, "absent_fp": absent},
        ]
        (folder / "frames.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    output = study.compare(tmp_path / "lean", tmp_path / "native")
    assert output["target_frames"] == 2 and output["empty_frames"] == 1
    assert output["lean_false_alarms"] == 1 and output["native_false_alarms"] == 0
    assert output["mean_delta"] == pytest.approx(0.2)
    (tmp_path / "native/seq3/inputs.json").write_text(json.dumps([
        {"frame": 0, "image_sha256": "different", "mask_sha256": "mask"}
    ]))
    with pytest.raises(ValueError, match="Unmatched input content"):
        study.compare(tmp_path / "lean", tmp_path / "native")
