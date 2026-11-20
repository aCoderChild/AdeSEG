"""Detector-initialized MedSAM2 video inference: box-only versus FGPI.

**No GT is read to choose prompts**. The detector's first eligible box is
frozen, the MedSAM2 Hiera-T/512 checkpoint supplies image features, and both
arms use the same propagation code. This is NOT the GT-mask oracle baseline.
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

from adeseg.prompts.feature_guided import feature_guided_prompt, apply_fgpi_prompt
from datasets.video import load_manifest, stage_frames
from modeling.medsam2_latest import (
    verify_medsam2_latest, build_verified_medsam2_latest, MEDSAM2_LATEST_SHA256,
)
from scripts.infer_med_video import sha256

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
                            mode="fgpi", label_id=1, confidence_gate=0.25,
                            min_feature_contrast=0.1):
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
            prompt=feature_guided_prompt(
                features, selected["box_xyxy"], hw,
                detector_confidence=selected["confidence"],
                min_detector_confidence=confidence_gate, mode=mode,
                min_feature_contrast=min_feature_contrast,
            )
            apply_fgpi_prompt(predictor,state,prompt_index,label_id,prompt)
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
        "prompt_source":"detector_box_with_optional_feature_points",
        "prompt_position":prompt_index, "prompt_frame_index":selected["frame_index"],
        "prompt_image_sha256":selected["image_sha256"],
        "prompt":prompt, "predictions":rows,
    }


def infer_detector_run(manifest, detections, detector_checkpoint, detector_sha256,
                       medsam2_checkpoint, output, *, split="dev", label_id=1,
                       mode="fgpi", device="cpu", confidence_gate=0.25,
                       min_feature_contrast=0.1):
    manifest=Path(manifest).resolve(strict=True)
    detections=Path(detections).resolve(strict=True)
    checkpoint=Path(detector_checkpoint).resolve(strict=True)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite previous experiment")
    if sha256(checkpoint) != detector_sha256:
        raise ValueError("Detector checkpoint SHA256 mismatch")
    verify_medsam2_latest(medsam2_checkpoint)
    videos=load_manifest(manifest,split,require_contiguous=True,require_jpeg=True,
                         safe_output_ids=True)
    records=load_detector_records(detections,videos)
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
            min_feature_contrast=min_feature_contrast)
    receipt={
        "protocol":PROTOCOL, "method":"FGPI", "mode":mode,
        "split":split, "label_id":label_id, "device":device,
        "manifest_sha256":sha256(manifest),
        "detections_sha256":sha256(detections),
        "detector_checkpoint_sha256":detector_sha256,
        "backbone_sha256":MEDSAM2_LATEST_SHA256,
        "model_config":MODEL_CONFIG, "postprocessing":"no_hole_filling",
        "confidence_gate":confidence_gate,
        "min_feature_contrast":min_feature_contrast,
        "prompt_budget":1,
        "uses_ground_truth_at_prompt_time":False,
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
    p.add_argument("--confidence-gate",type=float,default=0.25)
    p.add_argument("--min-feature-contrast",type=float,default=0.1)
    a=p.parse_args(argv)
    receipt=infer_detector_run(
        a.manifest,a.detections,a.detector_checkpoint,a.detector_sha256,
        a.checkpoint,a.output,split=a.split,label_id=a.label_id,
        mode=a.mode,device=a.device,confidence_gate=a.confidence_gate,
        min_feature_contrast=a.min_feature_contrast)
    print(json.dumps({"status":"generated_unverified_model_run","mode":a.mode,
                     "completed":sum(x["status"]=="completed" for x in receipt["results"].values())},indent=2))


if __name__=="__main__":
    main()
