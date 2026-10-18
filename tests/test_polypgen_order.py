import hashlib

import pytest

from scripts.build_polypgen_kaggle_manifest import verified_frame_order


def test_reindexed_files_require_provenance(tmp_path):
    files = [tmp_path / "0.jpg", tmp_path / "1.jpg"]
    with pytest.raises(ValueError, match="do not establish chronology"):
        verified_frame_order(files, "seq13")
    _, source = verified_frame_order(files, "seq13", allow_reindexed=True)
    assert source == "unverified_reindexed_diagnostic_only"


def test_original_numeric_suffix_order_unchanged(tmp_path):
    files = [tmp_path / f"EndoCV2021_seq13_{n}.jpg" for n in [100, 12, 0]]
    ordered, source = verified_frame_order(files, "seq13")
    assert [p.name for p in ordered] == [
        f"EndoCV2021_seq13_{n}.jpg" for n in [0, 12, 100]
    ]
    assert source == "original_numeric_suffix"


def test_mixed_names_cannot_bypass_reindexed_guard(tmp_path):
    with pytest.raises(ValueError, match="Mixed original and reindexed"):
        verified_frame_order(
            [tmp_path / "0.jpg", tmp_path / "original_12.jpg"], "seq13"
        )


def test_verified_mapping_binds_files_and_prevents_duplicate_frames(tmp_path):
    entries, files = {}, []
    for name, original in [("0.jpg", 100), ("1.jpg", 12)]:
        path = tmp_path / name
        path.write_bytes(name.encode())
        files.append(path)
        entries[name] = {
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "original_frame_index": original,
        }
    mapping = {"sequences": {"seq13": entries}}
    ordered, _ = verified_frame_order(files, "seq13", mapping)
    assert [p.name for p in ordered] == ["1.jpg", "0.jpg"]
    entries["1.jpg"]["original_frame_index"] = 100
    with pytest.raises(ValueError, match="one-to-one"):
        verified_frame_order(files, "seq13", mapping)
    entries["1.jpg"]["image_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="content mismatch"):
        verified_frame_order(files, "seq13", mapping)
