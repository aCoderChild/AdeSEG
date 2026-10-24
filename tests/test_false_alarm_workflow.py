import json
from pathlib import Path

import pytest

from scripts.backup_experiment_outputs import backup, digest
from scripts.run_stale_foreground import CONDITIONS, gate_suite_check, protection_check
from scripts.false_alarm_statistics import paired_method_summary


def test_backup_copies_bytes_and_symlinks_without_claiming_upload(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text('{"completed": true}')
    (source / "latest").symlink_to("result.json")
    target = tmp_path / "snapshot"
    backup(source, target)
    receipt = json.loads((target / "protection_receipt.json").read_text())
    assert receipt["backup_verified"]
    assert not receipt["drive_web_verified"]
    assert not receipt["independent_account_or_device"]
    assert (target / "latest").is_symlink()
    assert digest(target / "result.json") == digest(source / "result.json")
    assert receipt["backup_manifest_sha256"] == digest(target / "backup_manifest.json")
    with pytest.raises(ValueError, match="Drive-web"):
        protection_check(target / "protection_receipt.json")
    with pytest.raises(ValueError, match="new destination"):
        backup(source, target)
    with pytest.raises(ValueError, match="outside"):
        backup(source, source / "recursive")


def test_protection_requires_observed_url_and_unchanged_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    receipt = tmp_path / "receipt.json"
    data = dict(backup_verified=True, drive_web_verified=True,
                source_root=str(tmp_path), backup_manifest=str(manifest),
                backup_manifest_sha256=digest(manifest),
                drive_folder_url="https://drive.google.com/drive/folders/observed-id")
    receipt.write_text(json.dumps(data))
    assert protection_check(receipt)["backup_verified"]
    manifest.write_text("[1]")
    with pytest.raises(ValueError, match="checksum"):
        protection_check(receipt)


def test_all_four_gates_required_and_scientific_identity_must_match(tmp_path):
    for condition in CONDITIONS:
        folder = tmp_path / condition
        folder.mkdir()
        record = dict(passed=True, propagation_gate_verified=True,
                      gate_sequences=["seq4", "seq9"], signature=dict(
                          propagation_condition=condition, propagation_sources={"s": "sha"},
                          protection_receipt_sha256="proof", device="cpu",
                          checkpoint_sha256="checkpoint", manifest_sha256="manifest"))
        (folder / "validation.json").write_text(json.dumps(record))
        (folder / "delivery.json").write_text(json.dumps(dict(
            completed=True, wandb_artifact_upload_complete=True)))
    gate_suite_check(tmp_path, {"s": "sha"}, "proof")
    path = tmp_path / "samurai_fg_gate/validation.json"
    record = json.loads(path.read_text())
    record["signature"]["manifest_sha256"] = "wrong"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identities differ"):
        gate_suite_check(tmp_path, {"s": "sha"}, "proof")
    record["signature"]["manifest_sha256"] = "manifest"
    record["propagation_gate_verified"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Gate suite failed"):
        gate_suite_check(tmp_path, {"s": "sha"}, "proof")


def test_paired_summary_uses_sequence_not_frames_and_fixed_six_tests():
    def sequence(fp, dice, missed):
        return dict(gt_empty_false_alarm_rate=fp, present_dice=dice,
                    return_count=2, undetected_returns=missed)
    methods = {
        "sam2": {"a": sequence(.8, .7, 1), "b": sequence(.2, .8, 0)},
        "samurai": {"a": sequence(.6, .8, 0), "b": sequence(.2, .8, 0)},
        "samurai_fg_gate": {"a": sequence(.2, .7, 1), "b": sequence(.2, .8, 0)},
    }
    result = paired_method_summary(methods)
    assert len(result["comparisons"]) == 6
    row = result["comparisons"][3]
    assert row["endpoint"] == "gt_empty_false_alarm_rate"
    assert row["mean_difference"] == pytest.approx(-.2)
    assert row["n_sequences"] == 2
    assert all(0 <= row["holm6_p"] <= 1 for row in result["comparisons"])
