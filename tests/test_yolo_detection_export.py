"""YOLO detection exporter: frozen hashes, deterministic selection, no GT mask reads."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.export_yolo_detections import export_detections


class FakeYOLO:
    def __init__(self, bad=False):
        self.bad=bad
    def predict(self, **kwargs):
        boxes=torch.tensor([[3,3,13,13],[4,4,12,12],[2,2,8,8]],dtype=torch.float32)
        if self.bad:
            boxes[0,2]=99
        return [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=boxes,conf=torch.tensor([0.8,0.9,0.95]),
            cls=torch.tensor([1.,1.,2.])))]

def make_manifest(tmp_path):
    image=tmp_path/"0.jpg"
    mask=tmp_path/"0.png"
    Image.fromarray(np.zeros((16,16,3),np.uint8)).save(image)
    Image.fromarray(np.ones((16,16),np.uint8)).save(mask)
    manifest=tmp_path/"manifest.jsonl"
    manifest.write_text(json.dumps({
        "split":"dev","video_id":"case1","frame_index":0,
        "image":"0.jpg","mask":"0.png"
    })+"\n")
    model=tmp_path/"yolo.pt"
    model.write_bytes(b"mock-model")
    return manifest,model


def test_export_selects_highest_target_class_not_other_class(tmp_path):
    manifest,checkpoint=make_manifest(tmp_path)
    report=export_detections(
        manifest,FakeYOLO(),checkpoint,tmp_path/"detections.jsonl",target_class=1)
    saved=json.loads((tmp_path/"detections.jsonl").read_text())
    assert saved["confidence"]==pytest.approx(0.9)
    assert saved["box_xyxy"]==[4.0,4.0,12.0,12.0]
    assert report["detected_frames"]==1
    assert len(report["detector_checkpoint_sha256"])==64
    with pytest.raises(FileExistsError):
        export_detections(manifest,FakeYOLO(),checkpoint,
                          tmp_path/"detections.jsonl",target_class=1)


def test_invalid_yolo_geometry_is_rejected(tmp_path):
    manifest,checkpoint=make_manifest(tmp_path)
    with pytest.raises(ValueError,match="original image geometry"):
        export_detections(manifest,FakeYOLO(bad=True),checkpoint,
                          tmp_path/"bad.jsonl",target_class=1)


def test_no_target_class_creates_empty_frozen_detections(tmp_path):
    manifest,checkpoint=make_manifest(tmp_path)
    record=export_detections(manifest,FakeYOLO(),checkpoint,
                             tmp_path/"empty.jsonl",target_class=8)
    assert record["detected_frames"]==0
    assert (tmp_path/"empty.jsonl").read_text()==""
