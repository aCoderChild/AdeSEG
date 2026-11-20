"""Frozen YOLO manifest and model must match the exporter provenance."""
import json

import numpy as np
from PIL import Image
import pytest
import torch
from types import SimpleNamespace

from scripts.export_yolo_detections import export_detections
from scripts.detection_provenance import verify_detection_export


class Detector:
    def predict(self,**kwargs):
        return [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=torch.tensor([[2.,2.,6.,6.]]),
            cls=torch.tensor([1.]),conf=torch.tensor([0.75])))]

def fixtures(tmp_path):
    Image.fromarray(np.ones((8,8,3),np.uint8)).save(tmp_path/"0.jpg")
    Image.fromarray(np.zeros((8,8),np.uint8)).save(tmp_path/"0.png")
    manifest=tmp_path/"manifest.jsonl"
    manifest.write_text(json.dumps({
        "video_id":"v1","split":"dev","frame_index":0,
        "image":"0.jpg","mask":"0.png"
    })+"\n")
    checkpoint=tmp_path/"model.pt"
    checkpoint.write_bytes(b"trusted fake YOLO")
    detections=tmp_path/"detections.jsonl"
    export_detections(manifest,Detector(),checkpoint,detections,target_class=1)
    return manifest,checkpoint,detections

def test_export_metadata_is_authenticated(tmp_path):
    manifest,checkpoint,detections=fixtures(tmp_path)
    out=verify_detection_export(manifest,detections,checkpoint,split="dev")
    assert out["detected_frames"]==1
    assert len(out["metadata_sha256"])==64

def test_mutated_detections_cannot_reuse_old_metadata(tmp_path):
    manifest,checkpoint,detections=fixtures(tmp_path)
    detections.write_text(detections.read_text()+"\n")
    with pytest.raises(ValueError,match="metadata"):
        verify_detection_export(manifest,detections,checkpoint,split="dev")

def test_missing_metadata_requires_explicit_legacy_opt_in(tmp_path):
    manifest,checkpoint,detections=fixtures(tmp_path)
    detections.with_name(detections.name+".metadata.json").unlink()
    with pytest.raises(FileNotFoundError,match="Missing YOLO export metadata"):
        verify_detection_export(manifest,detections,checkpoint,split="dev")
