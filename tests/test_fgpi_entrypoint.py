"""CLI and authentication regressions for pre-experiment FGPI execution."""
import hashlib
import inspect
import json

import numpy as np
from PIL import Image
import pytest

import scripts.infer_fgpi_video as fgpi


def _assets(tmp_path):
    Image.fromarray(np.full((8,8,3),30,dtype=np.uint8)).save(tmp_path/"0.jpg")
    Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(tmp_path/"0.png")
    manifest=tmp_path/"manifest.jsonl"
    manifest.write_text(json.dumps({
        "split":"dev","video_id":"sample","frame_index":0,
        "image":"0.jpg","mask":"0.png"
    })+"\n")
    detector=tmp_path/"yolo.pt"
    detector.write_bytes(b"test fake detector")
    medsam=tmp_path/"medsam.pt"
    medsam.write_bytes(b"test fake MedSAM2")
    confidence=hashlib.sha256(detector.read_bytes()).hexdigest()
    detections=tmp_path/"boxes.jsonl"
    detections.write_text(json.dumps({
        "video_id":"sample","frame_index":0,"box_xyxy":[1,1,6,6],
        "confidence":0.8,
        "image_sha256":hashlib.sha256((tmp_path/"0.jpg").read_bytes()).hexdigest()
    })+"\n")
    return manifest, detections, detector, confidence, medsam


def test_opt_in_is_defined_on_run_function():
    params=inspect.signature(fgpi.infer_detector_run).parameters
    assert "allow_unverified_detections" in params
    assert params["allow_unverified_detections"].default is False
    assert "allow_unverified_detections" not in (
        inspect.signature(fgpi.infer_detector_sequence).parameters)


def test_cli_override_is_reachable_and_is_not_implicitly_enabled(tmp_path,monkeypatch):
    manifest,detections,detector,sha,medsam=_assets(tmp_path)
    received=[]
    def run(*args,**kwargs):
        received.append(kwargs["allow_unverified_detections"])
        return {"results":{}}
    monkeypatch.setattr(fgpi,"infer_detector_run",run)
    cli=[
        "--manifest",str(manifest),"--detections",str(detections),
        "--detector-checkpoint",str(detector),"--detector-sha256",sha,
        "--checkpoint",str(medsam),"--mode","fgpi","--output",str(tmp_path/"out")
    ]
    fgpi.main(cli)
    fgpi.main(cli+["--allow-unverified-detections"])
    assert received==[False,True]


def test_real_run_rejects_missing_metadata_without_opt_in(tmp_path,monkeypatch):
    manifest,detections,detector,sha,medsam=_assets(tmp_path)
    monkeypatch.setattr(fgpi,"verify_medsam2_latest",lambda _:None)
    with pytest.raises(FileNotFoundError,match="metadata"):
        fgpi.infer_detector_run(
            manifest,detections,detector,sha,medsam,tmp_path/"out",
            mode="fgpi",allow_unverified_detections=False)


def test_real_run_opt_in_is_explicitly_marked(tmp_path,monkeypatch):
    manifest,detections,detector,sha,medsam=_assets(tmp_path)
    monkeypatch.setattr(fgpi,"verify_medsam2_latest",lambda _:None)
    class FakeModel:
        fill_hole_area=0
        def eval(self):return self
        def requires_grad_(self,_):return self
    monkeypatch.setattr(fgpi,"build_verified_medsam2_latest",lambda *args,**kw:FakeModel())
    monkeypatch.setattr(fgpi,"infer_detector_sequence",lambda *args,**kw:{
        "status":"no_detector_prompt","frames":1,"predicted":0
    })
    out=fgpi.infer_detector_run(
        manifest,detections,detector,sha,medsam,tmp_path/"out",
        mode="fgpi",allow_unverified_detections=True)
    assert out["detector_export_verified"] is False
    assert out["detector_export_metadata_sha256"] is None
    assert (tmp_path/"out"/"run.json").exists()
