"""Basic fail-closed verification receipt and source fingerprint checks."""
from pathlib import Path
import pytest
from scripts.seal_med_video_study import collect_sha256, SOURCE_PATHS, ROOT, source_revision


def test_source_files_exist_and_are_tracked_by_receipt():
    assert len(SOURCE_PATHS) == len(set(SOURCE_PATHS))
    for name in SOURCE_PATHS:
        assert (ROOT / name).is_file(), name


def test_required_report_is_byte_hashed_and_missing_rejected(tmp_path):
    (tmp_path/"run.json").write_text('{"done":true}')
    result = collect_sha256(tmp_path, ["run.json"])
    assert len(result["run.json"]) == 64
    (tmp_path/"run.json").write_text('{"done":false}')
    assert collect_sha256(tmp_path, ["run.json"]) != result
    with pytest.raises(FileNotFoundError):
        collect_sha256(tmp_path, ["missing.json"])
