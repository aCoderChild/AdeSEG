"""GT-free Lean-SAM2 automatic prompt exporter and pinned-FPN contracts."""
import hashlib
import json

import numpy as np
from PIL import Image
import pytest
import torch

from datasets.video import ManifestFrame
from scripts.export_lean_fgpi_prompts import produce_sequence
from scripts.run_lean_memory_fusion_ablation import (
    initialize_frozen_lean_prompt, validate_direct_prompt,
)
from scripts.export_yolo_detections import export_detections
from scripts.detection_provenance import verify_detection_export


class FakeLeanPromptPredictor:
    def __init__(self, *, empty=False):
        self.empty = empty
        self.reset_calls = 0

    def init_state(self, **kwargs):
        return {"video_height": 32, "video_width": 32}

    def reset_state(self, state):
        self.reset_calls += 1

    def _get_image_feature(self, state, frame_idx, batch_size):
        assert frame_idx == 0 and batch_size == 1
        early = torch.ones(1, 32, 32, 32)
        early[:, 0, 8:24, 8:24] = 5
        mid = torch.ones(1, 64, 16, 16)
        deep = torch.ones(1, 256, 8, 8)
        return None, {"backbone_fpn": [early, mid, deep]}, None, None, None

    def add_new_points_or_box(self, state, **kwargs):
        self.received = kwargs
        values = torch.full((1, 1, 32, 32), -10.)
        if not self.empty:
            values[:, :, 10:20, 10:20] = 10
        return kwargs["frame_idx"], [kwargs["obj_id"]], values


def video(tmp_path):
    frames = []
    for i in range(3):
        path = tmp_path / f"{i}.jpg"
        Image.fromarray(np.ones((32, 32, 3), np.uint8) * 20).save(path)
        frames.append(ManifestFrame("seq", i, path, None))
    d = {
        "video_id": "seq", "frame_index": 1,
        "box_xyxy": [8, 8, 24, 24], "confidence": .8,
        "image_sha256": hashlib.sha256(frames[1].image_path.read_bytes()).hexdigest(),
    }
    return frames, d


def test_lean_fgpi_and_box_produce_compatible_frozen_mask_rows(tmp_path):
    frames, detection = video(tmp_path)
    fgpi, box = FakeLeanPromptPredictor(), FakeLeanPromptPredictor()
    r1, diag1 = produce_sequence(
        fgpi, frames, [detection], tmp_path / "fgpi", mode="fgpi")
    r2, diag2 = produce_sequence(
        box, frames, [detection], tmp_path / "box", mode="box_only")
    assert diag1["status"] == diag2["status"] == "completed"
    assert diag1["selected_box"] == diag2["selected_box"]
    assert r1["frame_index"] == r2["frame_index"] == 1
    assert r1["prompt_origin"] == r2["prompt_origin"] == "automatic_mask"
    assert fgpi.received["labels"] == [1, 0]
    assert box.received["labels"] is None
    assert fgpi.reset_calls == box.reset_calls == 1
    assert (tmp_path / "fgpi" / r1["mask_path"]).is_file()
    assert (tmp_path / "box" / r2["mask_path"]).is_file()
    assert r1["mask_sha256"] == r2["mask_sha256"]


def test_frozen_fgpi_points_reinitialize_native_sam_decoder_not_binary_mask(tmp_path):
    frames, detection = video(tmp_path)
    producer = FakeLeanPromptPredictor()
    prompt_row, _ = produce_sequence(
        producer, frames, [detection], tmp_path / "producer", mode="fgpi")
    assert prompt_row["direct_prompt"]["mode"] == "fgpi"
    assert validate_direct_prompt(
        prompt_row["direct_prompt"], frames[1].image_path)
    frozen = {
        **prompt_row,
        "resolved_mask": str(tmp_path / "producer" / prompt_row["mask_path"]),
    }
    direct = FakeLeanPromptPredictor()
    assert initialize_frozen_lean_prompt(
        direct, direct.init_state(), frozen,
        initialization_mode="direct_prompt") == "direct_prompt"
    assert direct.received["box"] == prompt_row["direct_prompt"]["box_xyxy"]
    assert direct.received["labels"] == [1, 0]
    bad = dict(frozen)
    bad["direct_prompt"] = {
        **frozen["direct_prompt"], "point_coords": [[9999, 0], [8, 8]],
    }
    with pytest.raises(ValueError, match="outside"):
        validate_direct_prompt(bad["direct_prompt"], frames[1].image_path)
    mismatched = FakeLeanPromptPredictor(empty=True)
    with pytest.raises(RuntimeError, match="no longer reproduces"):
        initialize_frozen_lean_prompt(
            mismatched, mismatched.init_state(), frozen,
            initialization_mode="direct_prompt")


def test_lean_fgpi_explicit_missing_and_empty_initialization(tmp_path):
    frames, d = video(tmp_path)
    assert produce_sequence(FakeLeanPromptPredictor(), frames, [],
                            tmp_path / "none")[1]["status"] == "no_detection"
    row, diagnostic = produce_sequence(
        FakeLeanPromptPredictor(empty=True), frames, [d], tmp_path / "empty")
    assert row is None
    assert diagnostic["status"] == "empty_initial_mask"


def test_lean_fgpi_rejects_gt_bearing_source_and_corrupted_fpn(tmp_path):
    frames, d = video(tmp_path)
    source_with_gt = [ManifestFrame(x.video_id, x.frame_index,
                                    x.image_path, tmp_path / "a.png")
                      for x in frames]
    with pytest.raises(ValueError, match="GT-bearing"):
        produce_sequence(FakeLeanPromptPredictor(), source_with_gt,
                         [d], tmp_path / "gt")
    class BadFPN(FakeLeanPromptPredictor):
        def _get_image_feature(self, state, frame_idx, batch_size):
            return None, {"backbone_fpn": [torch.ones(1, 256, 8, 8)]}, None, None, None
    with pytest.raises(RuntimeError, match="32/64/256"):
        produce_sequence(BadFPN(), frames, [d], tmp_path / "bad")


class FakeYOLO:
    def predict(self, **kwargs):
        from types import SimpleNamespace
        return [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=torch.tensor([[8., 8., 24., 24.]]),
            cls=torch.tensor([1.]), conf=torch.tensor([.8])))]

def test_gt_free_yolo_export_and_metadata_are_strict(tmp_path):
    frames, d = video(tmp_path)
    manifest = tmp_path / "images.jsonl"
    manifest.write_text("".join(json.dumps({
        "split": "train", "video_id": x.video_id,
        "frame_index": x.frame_index,
        "image": x.image_path.name,
    }) + "\n" for x in frames))
    checkpoint = tmp_path / "yolo.pt"
    checkpoint.write_bytes(b"fake detector")
    detections = tmp_path / "detections.jsonl"
    report = export_detections(
        manifest, FakeYOLO(), checkpoint, detections,
        split="train", target_class=1, image_only_manifest=True)
    assert report["image_only_manifest"] is True
    verified = verify_detection_export(
        manifest, detections, checkpoint, split="train",
        image_only_manifest=True)
    assert verified["detected_frames"] == len(frames)
    with pytest.raises(ValueError, match="metadata"):
        verify_detection_export(manifest, detections, checkpoint, split="train")
