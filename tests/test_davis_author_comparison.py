import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import run_davis_author_vos_comparator as author_run
from scripts import summarize_davis_region_vs_selection as summarize
from scripts import queue_davis_region_vs_temporal_selection as queue
from scripts.memory_token_io import digest


def test_author_methods_use_their_actual_cli_entry_points():
    samurai = SimpleNamespace(vos_inference=lambda **kwargs: None)
    sam2long = SimpleNamespace(vos_separate_inference_per_object=lambda **kwargs: None)
    assert author_run.author_vos_function("samurai", samurai) is samurai.vos_inference
    assert author_run.author_vos_function("sam2long", sam2long) is sam2long.vos_separate_inference_per_object


def test_gate_selects_only_the_two_object_engineering_video():
    records = [
        {"video": "another-video", "object_ids": [1]},
        {"video": "bmx-bumps", "object_ids": [1, 2]},
    ]
    assert author_run.videos_for_stage(records, "gate") == [records[1]]
    assert author_run.videos_for_stage(records, "full") == records

    records[1]["object_ids"] = [1]
    with pytest.raises(RuntimeError, match="Two-object train engineering video changed"):
        author_run.videos_for_stage(records, "gate")


def test_queue_protocol_snapshot_is_immutable(tmp_path):
    protocol = tmp_path / "protocol.json"
    snapshot = tmp_path / "drive_protocol.json"
    protocol.write_text('{"version": 1}\n')
    snapshot.write_bytes(protocol.read_bytes())
    frozen_sha = digest(protocol)
    queue.verify_frozen_protocol(protocol, snapshot, frozen_sha)

    protocol.write_text('{"version": 2}\n')
    with pytest.raises(RuntimeError, match="protocol changed"):
        queue.verify_frozen_protocol(protocol, snapshot, frozen_sha)


def test_local_reference_requires_same_setup_and_complete_results(tmp_path, monkeypatch):
    local_run = tmp_path / "local"
    local_run.mkdir()
    checkpoint = tmp_path / "tiny.pt"
    checkpoint.write_bytes(b"checkpoint")
    inventory = {
        "dataset": "davis2017", "split": "val", "video_count": 1,
        "videos": [{"video": "v", "object_ids": [1]}],
    }
    monkeypatch.setattr(author_run, "inventory", lambda *args: inventory)
    inventory_path = local_run / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    setup = {
        "dataset": "davis2017", "split": "val",
        "checkpoint_sha256": digest(checkpoint),
        "model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "indexed_mask_non_overlap_outputs": True,
        "dataset_inventory_sha256": digest(inventory_path),
    }
    (local_run / "setup.json").write_text(json.dumps(setup))
    conditions = {
        name: {"official_evaluation": {"per_video": {"v": {"J&F-Mean": 0.5}}}}
        for name in ("native", "generic_local_tome", "soft_mask_aware_tome")
    }
    (local_run / "results.json").write_text(json.dumps({"conditions": conditions}))
    result = author_run.validate_local_reference(local_run, tmp_path, checkpoint)
    assert result["setup_sha256"] == digest(local_run / "setup.json")

    setup["checkpoint_sha256"] = "wrong"
    (local_run / "setup.json").write_text(json.dumps(setup))
    with pytest.raises(RuntimeError, match="different checkpoint"):
        author_run.validate_local_reference(local_run, tmp_path, checkpoint)


def test_gate_evaluator_is_limited_to_the_gate_video(tmp_path, monkeypatch):
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def evaluate(self, prediction_dir, metric):
            assert metric == ("J", "F")
            return {"accepted": True}

    package = types.ModuleType("davis2017")
    package.__path__ = []
    evaluator_module = types.ModuleType("davis2017.evaluation")
    evaluator_module.DAVISEvaluation = FakeEvaluator
    fake_summary = types.ModuleType("scripts.run_davis_soft_mask_aware_tome")
    fake_summary.summarize_official_metrics = lambda metrics, records: {
        "accepted": metrics["accepted"], "videos": [row["video"] for row in records]
    }
    monkeypatch.setattr(author_run, "source_revision", lambda path: author_run.EVALUATOR_REVISION)
    monkeypatch.setitem(sys.modules, "davis2017", package)
    monkeypatch.setitem(sys.modules, "davis2017.evaluation", evaluator_module)
    monkeypatch.setitem(sys.modules, "scripts.run_davis_soft_mask_aware_tome", fake_summary)

    records = [{"video": "bmx-bumps"}]
    result = author_run.evaluator_summary(tmp_path, tmp_path, records, "train")
    assert captured["gt_set"] == "train"
    assert captured["sequences"] == ["bmx-bumps"]
    assert result == {"accepted": True, "videos": ["bmx-bumps"]}


def test_prediction_validation_rejects_reserved_void_label(tmp_path, monkeypatch):
    mask_root = tmp_path / "Annotations" / "480p"
    (mask_root / "v").mkdir(parents=True)
    prediction_dir = tmp_path / "predictions"
    (prediction_dir / "v").mkdir(parents=True)
    prompt = np.zeros((4, 4), dtype=np.uint8)
    prompt[1, 1] = 1
    later = prompt.copy()
    later[0, 0] = 255
    Image.fromarray(prompt).save(mask_root / "v" / "00000.png")
    Image.fromarray(prompt).save(mask_root / "v" / "00001.png")
    Image.fromarray(prompt).save(prediction_dir / "v" / "00000.png")
    Image.fromarray(later).save(prediction_dir / "v" / "00001.png")
    monkeypatch.setattr(author_run, "layout", lambda *args: (None, mask_root, None, None))
    record = {
        "frames": ["00000.jpg", "00001.jpg"],
        "height": 4,
        "width": 4,
        "object_ids": [1],
    }
    with pytest.raises(RuntimeError, match="Unexpected object IDs"):
        author_run.validate_prediction_folder(prediction_dir, "v", record, tmp_path, "val")


def test_saved_prediction_verification_checks_hashes(tmp_path):
    prediction_dir = tmp_path / "predictions"
    (prediction_dir / "v").mkdir(parents=True)
    path = prediction_dir / "v" / "00000.png"
    path.write_bytes(b"mask")
    records = [{"video": "v", "frames": ["00000.png"]}]
    receipts = {"v": {"video": "v", "frames": 1,
                       "prediction_sha256": {"00000.png": digest(path)}}}
    summarize.verify_predictions(prediction_dir, records, receipts)
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed saved prediction"):
        summarize.verify_predictions(prediction_dir, records, receipts)


def test_davis_framewise_dice_uses_only_official_scored_frames(tmp_path):
    root = tmp_path / "DAVIS"
    mask_dir = root / "Annotations" / "480p" / "v"
    pred_dir = tmp_path / "predictions" / "v"
    mask_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    empty = np.zeros((4, 4), dtype=np.uint8)
    gt = empty.copy()
    gt[1:3, 1:3] = 1
    for name, mask in (("00000.png", gt), ("00001.png", gt), ("00002.png", gt)):
        Image.fromarray(mask).save(mask_dir / name)
        Image.fromarray(mask).save(pred_dir / name)
    record = {"video": "v", "frames": ["00000.png", "00001.png", "00002.png"],
              "object_ids": [1]}
    scores = summarize.framewise_dice(tmp_path / "predictions", root, [record], "val")
    assert scores["global_object_macro_mean"] == 1.0
    assert scores["per_video"] == {"v": 1.0}


def test_paired_contrasts_use_video_as_the_unit():
    names = ("native", "generic_local_tome", "soft_mask_aware_tome", "samurai", "sam2long")
    scores = {
        name: {"per_video": {f"v{i}": {"J&F-Mean": 0.5 + offset + i * 0.01}
                              for i in range(4)}}
        for name, offset in zip(names, (0.0, 0.01, 0.03, -0.01, -0.02))
    }
    contrasts = summarize.paired_contrasts(scores)
    assert contrasts["soft_mask_aware_tome - native"]["n_videos"] == 4
    assert contrasts["soft_mask_aware_tome - native"]["wins_ties_losses"] == [4, 0, 0]
    assert all("holm_adjusted_p" in row for row in contrasts.values())
