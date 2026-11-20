"""Detector-initialized MedSAM2 video inference: box-only versus FGPI.

**No GT is read to choose prompts**. The detector's first eligible box is
frozen, the MedSAM2 Hiera-T/512 checkpoint supplies image features, and both
arms use the same propagation code. FGPI always supplies box + positive and
negative feature points, without a feature-reliability fallback. box_only
is the matched control. This is NOT the GT-mask oracle baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from adeseg.modules.fgpi import FGPI, FGPIConfig
from adeseg.prompts.feature_guided import feature_guided_prompt
from adeseg.prompts.target_anchored import target_anchored_points
from datasets.video import load_manifest, stage_frames
from modeling.medsam2_latest import (
    verify_medsam2_latest, build_verified_medsam2_latest, MEDSAM2_LATEST_SHA256,
)
from scripts.infer_med_video import sha256
from scripts.detection_provenance import verify_detection_export

MODEL_CONFIG = "sam2.1_hiera_t512.yaml"
PROTOCOL = "detector_first_box_post_prompt_v1"


def load_detector_records(path, videos):
    """Pin detector outputs to exact source images without consulting masks."""
    path = Path(path).resolve(strict=True)
    mapping, seen = {}, set()
    source_indexes = {
        video: {frame.frame_index: frame for frame in frames}
        for video, frames in videos.items()
    }
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or set(record) != {
            "video_id", "frame_index", "box_xyxy", "confidence", "image_sha256"
        }:
            raise ValueError(f"Detection row {number}: unexpected or missing fields")
        name, index = record["video_id"], record["frame_index"]
        if (name not in source_indexes or type(index) is not int
                or index not in source_indexes[name] or (name, index) in seen):
            raise ValueError(f"Detection row {number}: unknown or duplicate video frame")
        seen.add((name,index))
        if record["image_sha256"] != sha256(source_indexes[name][index].image_path):
            raise ValueError(f"Detection source image SHA256 mismatch: {name}:{index}")
        if (type(record["confidence"]) not in (int,float)
                or not 0 <= record["confidence"] <= 1):
            raise ValueError("Detector confidence must be in [0,1]")
        # Geometry is checked again by the FGPI module against actual video size.
        mapping.setdefault(name,[]).append(record)
    for name in mapping:
        mapping[name].sort(key=lambda row: row["frame_index"])
    return mapping


def infer_detector_sequence(predictor, frames, records, output, *,
                            mode="fgpi", label_id=1, confidence_gate=0.0,
                            positives=1, negatives=1, max_anchors=4):
    """One detector prompt per sequence; subsequent frames receive NO GT or clicks."""
    valid = [r for r in records if r["confidence"] >= confidence_gate]
    if not valid:
        return {"status":"no_detector_prompt", "frames":len(frames), "predicted":0}
    selected = valid[0]
    positions = {frame.frame_index:i for i,frame in enumerate(frames)}
    prompt_index = positions[selected["frame_index"]]
    output_dir = output / "predictions" / frames[0].video_id
    output_dir.mkdir(parents=True, exist_ok=False)
    rows=[]
    with tempfile.TemporaryDirectory(prefix="adeseg_fgpi_") as staging:
        stage_frames(frames,Path(staging))
        with torch.inference_mode():
            state=predictor.init_state(video_path=staging,
                                       offload_video_to_cpu=True,
                                       offload_state_to_cpu=False)
            hw = (int(state["video_height"]),int(state["video_width"]))
            features=None
            if mode == "fgpi":
                _, backbone, *_ = predictor._get_image_feature(
                    state, frame_idx=prompt_index, batch_size=1)
                early=backbone.get("backbone_fpn")
                if not isinstance(early,(tuple,list)) or not early:
                    raise RuntimeError("Native MedSAM2 early FPN features unavailable")
                features=early[0]
            if mode == "fgpi":
                prompt = FGPI(FGPIConfig(
                    positive_points=positives,
                    negative_points=negatives,
                    max_pseudo_anchors=max_anchors,
                )).construct(
                    features, selected["box_xyxy"], hw,
                    detector_confidence=selected["confidence"],
                )
            elif mode == "box_only":
                prompt = feature_guided_prompt(
                    None, selected["box_xyxy"], hw,
                    detector_confidence=selected["confidence"],
                    min_detector_confidence=confidence_gate, mode="box_only",
                )
            else:
                raise ValueError("Expected fgpi or box_only prompt mode")
            init_index, init_ids, init_logits = FGPI.initialize(
                predictor,state,prompt_index,label_id,prompt)
            if init_index != prompt_index or list(init_ids) != [label_id]:
                raise RuntimeError("SAM2 initialization returned a different object/frame")
            if tuple(init_logits.shape) != (1,1,*hw):
                raise ValueError("SAM2 initialization logit geometry differs")
            init_values=init_logits[0,0].detach().float().cpu().contiguous().numpy()
            if not np.isfinite(init_values).all():
                raise FloatingPointError("Nonfinite SAM2 initialization mask")
            init_path=output/"initialization"/frames[0].video_id/(
                f"{selected['frame_index']:06d}.png")
            init_path.parent.mkdir(parents=True,exist_ok=True)
            if init_path.exists():
                raise FileExistsError(init_path)
            Image.fromarray(np.uint8(init_values>0)*255).save(init_path)
            initial_record={
                "frame_index":selected["frame_index"],
                "source_image_sha256":selected["image_sha256"],
                "source_gt_sha256":sha256(frames[prompt_index].mask_path),
                "prediction_file":str(init_path.relative_to(output)),
                "prediction_sha256":sha256(init_path),
                "logit_sha256":hashlib.sha256(
                    np.ascontiguousarray(init_values,dtype=np.float32).tobytes()
                ).hexdigest(),
            }
            for index, object_ids, logits in predictor.propagate_in_video(
                state,start_frame_idx=prompt_index,reverse=False
            ):
                if list(object_ids) != [label_id]:
                    raise RuntimeError("Unexpected SAM2 object IDs")
                if index == prompt_index:
                    continue
                if index != prompt_index + len(rows)+1:
                    raise RuntimeError("Nonconsecutive or duplicate prediction")
                expected_hw=(int(state["video_height"]),int(state["video_width"]))
                if tuple(logits.shape) != (1,1,*expected_hw):
                    raise ValueError("SAM2 logit geometry differs from source video")
                values=logits[0,0].detach().float().cpu().contiguous().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite detector-prompted SAM2 logits")
                path=output_dir/f"{frames[index].frame_index:06d}.png"
                if path.exists():
                    raise FileExistsError(path)
                Image.fromarray(np.uint8(values > 0)*255).save(path)
                rows.append({
                    "frame_index": frames[index].frame_index,
                    "source_image_sha256":sha256(frames[index].image_path),
                    "source_gt_sha256":sha256(frames[index].mask_path),
                    "prediction_file":str(path.relative_to(output)),
                    "prediction_sha256":sha256(path),
                    "logit_sha256":hashlib.sha256(
                        np.ascontiguousarray(values,dtype=np.float32).tobytes()
                    ).hexdigest(),
                })
    if len(rows) != len(frames)-prompt_index-1:
        raise RuntimeError("Incomplete detector-prompted propagation")
    return {
        "status":"completed", "frames":len(frames), "predicted":len(rows),
        "prompt_source":("detector_box_plus_feature_points" if mode == "fgpi"
                         else "detector_box_only"),
        "prompt_position":prompt_index, "prompt_frame_index":selected["frame_index"],
        "prompt_image_sha256":selected["image_sha256"],
        "prompt":prompt, "initialization":initial_record, "predictions":rows,
    }


def infer_detector_run(manifest, detections, detector_checkpoint, detector_sha256,
                       medsam2_checkpoint, output, *, split="dev", label_id=1,
                       mode="fgpi", device="cpu", confidence_gate=0.0,
                       positives=1, negatives=1, max_anchors=4,
                       allow_unverified_detections=False):
    manifest=Path(manifest).resolve(strict=True)
    detections=Path(detections).resolve(strict=True)
    checkpoint=Path(detector_checkpoint).resolve(strict=True)
    output=Path(output).resolve()
    if (type(positives) is not int or type(negatives) is not int
            or not (1 <= positives <= 8 and 1 <= negatives <= 8)
            or type(max_anchors) is not int or not 1 <= max_anchors <= 16):
        raise ValueError("FGPI requires positive, negative and anchor budgets")
    if output.exists():
        raise FileExistsError("Refusing to overwrite previous experiment")
    if type(allow_unverified_detections) is not bool:
        raise ValueError("Unverified detector override must be boolean")
    if sha256(checkpoint) != detector_sha256:
        raise ValueError("Detector checkpoint SHA256 mismatch")
    verify_medsam2_latest(medsam2_checkpoint)
    videos=load_manifest(manifest,split,require_contiguous=True,require_jpeg=True,
                         safe_output_ids=True)
    records=load_detector_records(detections,videos)
    try:
        detector_evidence=verify_detection_export(
            manifest,detections,checkpoint,split=split)
    except FileNotFoundError:
        if not allow_unverified_detections:
            raise
        detector_evidence=None
    predictor=build_verified_medsam2_latest(
        medsam2_checkpoint,model_cfg=MODEL_CONFIG,device=device,
        apply_postprocessing=False).eval().requires_grad_(False)
    predictor.fill_hole_area=0
    output.mkdir(parents=True)
    results={}
    for name,frames in sorted(videos.items()):
        results[name]=infer_detector_sequence(
            predictor,frames,records.get(name,[]),output,mode=mode,
            label_id=label_id,confidence_gate=confidence_gate,
            positives=positives,negatives=negatives,max_anchors=max_anchors)
    receipt={
        "protocol":PROTOCOL, "method":"FGPI", "mode":mode,
        "split":split, "label_id":label_id, "device":device,
        "manifest_sha256":sha256(manifest),
        "detections_sha256":sha256(detections),
        "detector_checkpoint_sha256":detector_sha256,
        "detector_export_verified": detector_evidence is not None,
        "detector_export_metadata_sha256": (
            detector_evidence["metadata_sha256"] if detector_evidence is not None else None),
        "backbone_sha256":MEDSAM2_LATEST_SHA256,
        "model_config":MODEL_CONFIG, "postprocessing":"no_hole_filling",
        "confidence_gate":confidence_gate,
        "prompt_budget":1,
        "point_budget":{"positives":positives,"negatives":negatives,
                        "max_anchors":max_anchors},
        "prompt_selector_source_sha256":sha256(
            Path(target_anchored_points.__code__.co_filename)),
        "uses_ground_truth_at_prompt_time":False,
        "initialization_prediction_scored":True,
        "results":results,
    }
    (output/"run.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--detections",type=Path,required=True)
    p.add_argument("--detector-checkpoint",type=Path,required=True)
    p.add_argument("--detector-sha256",required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=("dev","test"),default="dev")
    p.add_argument("--mode",choices=("box_only","fgpi"),required=True)
    p.add_argument("--label-id",type=int,default=1)
    p.add_argument("--device",choices=("cpu","mps","cuda"),default="cpu")
    p.add_argument("--confidence-gate",type=float,default=0.0,
                   help="Optional upstream YOLO filtering (default 0: use every provided detection)")
    p.add_argument("--positives",type=int,default=1,
                   help="Fixed number of FGPI positive points (default 1)")
    p.add_argument("--negatives",type=int,default=1,
                   help="Fixed number of FGPI background points (default 1)")
    p.add_argument("--max-anchors",type=int,default=4,
                   help="Maximum diverse box-derived target anchor tokens (default 4)")
    p.add_argument("--allow-unverified-detections",action="store_true",
                   help="Exploratory legacy JSONL only; forfeits YOLO exporter provenance")
    a=p.parse_args(argv)
    receipt=infer_detector_run(
        a.manifest,a.detections,a.detector_checkpoint,a.detector_sha256,
        a.checkpoint,a.output,split=a.split,label_id=a.label_id,
        mode=a.mode,device=a.device,confidence_gate=a.confidence_gate,
        positives=a.positives,negatives=a.negatives,max_anchors=a.max_anchors,
        allow_unverified_detections=a.allow_unverified_detections)
    print(json.dumps({"status":"generated_unverified_model_run","mode":a.mode,
                     "completed":sum(x["status"]=="completed" for x in receipt["results"].values())},indent=2))


if __name__=="__main__":
    main()
