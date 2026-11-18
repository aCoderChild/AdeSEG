"""Fail-closed, source-group-level partition validation for new TRAIN runs.

A manifest field saying "split": "train" alone cannot establish independence.
Provide a separately frozen source-group partition and per-capture SHA256.
This validates declarations and bytes; independently verifying that groups
map to unique patients still requires original dataset provenance.
"""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_train_sources(manifest_path, split_file, *, file_field):
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    split_file = Path(split_file).expanduser().resolve(strict=True)
    splits = json.loads(split_file.read_text(encoding="utf-8"))
    if not isinstance(splits, dict) or set(splits) != {"train", "dev", "test"}:
        raise ValueError("Split file must contain train, dev and test groups")
    groups = {}
    for name, values in splits.items():
        if (not isinstance(values, list) or
                any(not isinstance(v, str) or not v.strip() for v in values) or
                len(values) != len(set(values))):
            raise ValueError("Each split must contain unique nonempty source groups")
        groups[name] = set(values)
    if not groups["train"]:
        raise ValueError("No training source groups")
    if any(groups[a] & groups[b] for a, b in
           (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Overlapping patient/source groups across splits")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Training manifest must be a nonempty array")
    checksums = []
    seen = set()
    for row in records:
        if not isinstance(row, dict) or row.get("split") != "train":
            raise ValueError("A non-training row was found in a training manifest")
        group = row.get("source_group")
        if not isinstance(group, str) or group not in groups["train"]:
            raise ValueError("Capture source_group not in sealed train partition")
        name = row.get(file_field)
        if not isinstance(name, str) or not name.endswith(".pt"):
            raise ValueError("Capture file name missing or not a .pt file")
        path = (manifest_path.parent / name).resolve(strict=True)
        if not path.is_file() or path in seen:
            raise ValueError("Duplicate or missing capture file")
        seen.add(path)
        expected = row.get("capture_sha256")
        if (not isinstance(expected, str) or len(expected) != 64 or
                any(c not in "0123456789abcdef" for c in expected) or
                expected != sha256(path)):
            raise ValueError("Training capture SHA256 mismatch")
        checksums.append(expected)
    return {
        "split_file_sha256": sha256(split_file),
        "manifest_sha256": sha256(manifest_path),
        "capture_count": len(checksums),
        "train_source_groups": sorted({row["source_group"] for row in records}),
        "validated_capture_sha256": checksums,
        "note": "Manifest groups are asserted, not independently verified patient identities",
    }
