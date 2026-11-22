"""Authenticate a frozen YOLO detection export before publication-grade FGPI runs."""
from __future__ import annotations

import json
from pathlib import Path

from datasets.video import load_manifest
from scripts.infer_med_video import sha256


def verify_detection_export(manifest, detections, detector_checkpoint, *,
                            split, image_only_manifest=False):
    manifest=Path(manifest).resolve(strict=True)
    detections=Path(detections).resolve(strict=True)
    checkpoint=Path(detector_checkpoint).resolve(strict=True)
    if type(image_only_manifest) is not bool:
        raise ValueError("image_only_manifest must be boolean")
    metadata=detections.with_name(detections.name+".metadata.json")
    if not metadata.is_file():
        raise FileNotFoundError(
            f"Missing YOLO export metadata: {metadata}. "
            "Re-export with scripts.export_yolo_detections, or explicitly "
            "approve unverified legacy detections."
        )
    row=json.loads(metadata.read_text(encoding="utf-8"))
    required={
        "schema":"adeseg_fgpi_yolo_single_box_per_frame_v1",
        "manifest_sha256":sha256(manifest),
        "detections_sha256":sha256(detections),
        "detector_checkpoint_sha256":sha256(checkpoint),
        "split":split,
        "ground_truth_used_in_detection":False,
        "image_only_manifest": bool(image_only_manifest),
        "policy":"top-confidence target-class YOLO detection per frame",
    }
    if not isinstance(row,dict) or any(row.get(k)!=v for k,v in required.items()):
        raise ValueError("YOLO export metadata does not match authenticated study inputs")
    if (type(row.get("target_class")) is not int or row["target_class"]<0
            or type(row.get("detected_frames")) is not int
            or type(row.get("total_frames")) is not int
            or type(row.get("image_size")) is not int
            or not 0<=row.get("yolo_conf",-1)<=1
            or not 0<row.get("iou",0)<=1):
        raise ValueError("Invalid frozen YOLO method parameters")
    videos=load_manifest(manifest,split,require_contiguous=True,
                         require_jpeg=True,safe_output_ids=True,
                         require_masks=not image_only_manifest)
    expected_total=sum(len(frames) for frames in videos.values())
    lines=[r for r in detections.read_text(encoding="utf-8").splitlines() if r.strip()]
    if row["total_frames"]!=expected_total or row["detected_frames"]!=len(lines):
        raise ValueError("Frozen YOLO export frame population differs")
    return {"metadata_path":metadata,"metadata_sha256":sha256(metadata),
            "target_class":row["target_class"],"detected_frames":len(lines),
            "total_frames":expected_total}
