"""Score automatic FGPI and matched detector-box runs without oracle prompting.

For each detector-seeded sequence the prompt frame is EXCLUDED, post-prompt
GT-present Dice and GT-empty false alarms are reported separately. Sequences
without a detector prompt are visible exclusions, not missing zeros.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from evaluation.temporal_consistency import adjacent_mask_metrics, undefined_temporal_metrics
from scripts.infer_med_video import sha256
from scripts.infer_fgpi_video import PROTOCOL, load_detector_records
from scripts.detection_provenance import verify_detection_export
from adeseg.prompts.target_anchored import target_anchored_points


def evaluate_fgpi_run(manifest, detections, detector_checkpoint, run):
    manifest = Path(manifest).resolve(strict=True)
    detections = Path(detections).resolve(strict=True)
    detector_checkpoint = Path(detector_checkpoint).resolve(strict=True)
    root = Path(run).resolve(strict=True)
    receipt = json.loads((root/"run.json").read_text(encoding="utf-8"))
    if (receipt.get("protocol") != PROTOCOL
            or receipt.get("mode") not in ("box_only", "fgpi")
            or receipt.get("uses_ground_truth_at_prompt_time") is not False):
        raise ValueError("Not a detector-only FGPI experiment")
    for field, path in (("manifest_sha256",manifest),
                        ("detections_sha256",detections),
                        ("detector_checkpoint_sha256",detector_checkpoint)):
        if receipt.get(field) != sha256(path):
            raise ValueError(f"FGPI {field} mismatch")
    budget=receipt.get("point_budget")
    if (not isinstance(budget,dict) or set(budget)!=
            {"positives","negatives","max_anchors"}
            or any(type(v) is not int for v in budget.values())
            or not (1 <= budget["positives"] <= 8 and
                    1 <= budget["negatives"] <= 8 and
                    1 <= budget["max_anchors"] <= 16)):
        raise ValueError("FGPI receipt lacks predeclared point/anchor budgets")
    if receipt.get("prompt_selector_source_sha256") != sha256(
            Path(target_anchored_points.__code__.co_filename)):
        raise ValueError("FGPI prompt selector source SHA256 differs")
    if receipt.get("detector_export_verified") is True:
        evidence=verify_detection_export(manifest,detections,detector_checkpoint,
                                         split=receipt["split"])
        if evidence["metadata_sha256"]!=receipt.get("detector_export_metadata_sha256"):
            raise ValueError("YOLO exporter lineage changed")
    elif receipt.get("detector_export_verified") is not False:
        raise ValueError("FGPI detector export verification status missing")
    label_id=receipt["label_id"]
    videos=load_manifest(manifest,receipt["split"],require_contiguous=True,
                         require_jpeg=True,safe_output_ids=True)
    detected=load_detector_records(detections,videos)
    if set(videos) != set(receipt["results"]):
        raise ValueError("Detector run changed its sequence population")
    frames_all, sequences, excluded=[],[],[]
    initialization_rows=[]
    for name,frames in sorted(videos.items()):
        record=receipt["results"][name]
        elig=[r for r in detected.get(name,[])
              if r["confidence"] >= receipt["confidence_gate"]]
        if not elig:
            if record.get("status") != "no_detector_prompt" or record.get("predicted") != 0:
                raise ValueError("Detection-free sequence was not excluded")
            excluded.append(name)
            continue
        if record.get("status") != "completed":
            raise ValueError("Eligible video has no completed predictions")
        first=elig[0]
        source_idx=[f.frame_index for f in frames].index(first["frame_index"])
        prompt=record["prompt"]
        if (record.get("prompt_position") != source_idx
                or record.get("prompt_frame_index") != first["frame_index"]
                or record.get("prompt_image_sha256") != first["image_sha256"]
                or prompt.get("box_xyxy") != [float(x) for x in first["box_xyxy"]]
                or prompt.get("mode") != receipt["mode"]
                or prompt.get("detector_confidence") != first["confidence"]
                or record.get("prompt_source") != (
                    "detector_box_plus_feature_points" if receipt["mode"] == "fgpi"
                    else "detector_box_only")):
            raise ValueError("Detector prompt differs from sealed box or frame")
        if receipt["mode"] == "box_only" and (prompt["point_coords"] or prompt["point_labels"]):
            raise ValueError("Box-only arm unexpectedly contains feature points")
        if receipt["mode"] == "fgpi" and (
            prompt.get("decision") != "augmented"
            or prompt.get("selection_strategy") !=
                "box_pseudo_anchor_cosine_bg_contrast_spatial_v1"
            or prompt.get("reason") != "mandatory_target_anchored_feature_points"
            or prompt["point_labels"].count(1) != budget["positives"]
            or prompt["point_labels"].count(0) != budget["negatives"]
            or type(prompt.get("anchor_count")) is not int
            or not (1 <= prompt["anchor_count"] <= budget["max_anchors"])
            or len(prompt.get("anchor_coords", [])) != prompt["anchor_count"]
        ):
            raise ValueError("FGPI run lacks its required target anchors or fixed point budget")
        if (len(prompt["point_coords"]) != len(prompt["point_labels"])
                or any(x not in (0,1) for x in prompt["point_labels"])
                or len({tuple(pt) for pt in prompt["point_coords"]}) != len(prompt["point_coords"])):
            raise ValueError("Invalid or duplicate FGPI point declaration")
        if receipt["mode"] == "fgpi":
            x0,y0,x1,y1=prompt["box_xyxy"]
            for (x,y),label in zip(prompt["point_coords"],prompt["point_labels"]):
                inside=(x0 <= x <= x1 and y0 <= y <= y1)
                if (label==1 and not inside) or (label==0 and inside):
                    raise ValueError("FGPI positive/negative points violate detector box geometry")
        # Evaluate the actual initialization-frame prediction separately.
        # This score is never used to select the prompt or to rerank tokens.
        init=record.get("initialization")
        expected_init=Path("initialization")/name/f"{first['frame_index']:06d}.png"
        if (not isinstance(init,dict)
                or init.get("frame_index")!=first["frame_index"]
                or init.get("source_image_sha256")!=first["image_sha256"]
                or init.get("source_gt_sha256")!=sha256(frames[source_idx].mask_path)
                or init.get("prediction_file")!=expected_init.as_posix()):
            raise ValueError("Missing or mismatched FGPI initialization prediction")
        init_path=(root/expected_init).resolve(strict=True)
        if not init_path.is_relative_to(root) or sha256(init_path)!=init.get("prediction_sha256"):
            raise ValueError("FGPI initial prediction SHA256 mismatch")
        with Image.open(init_path) as image:
            initial_prediction=np.asarray(image.convert("L"))>=128
        initial_gt=load_label_mask(frames[source_idx].mask_path,[label_id])==label_id
        if initial_prediction.shape!=initial_gt.shape:
            raise ValueError("FGPI initial prediction shape mismatch")
        initial_scores=calculate_scores(initial_prediction,initial_gt)
        initialization_rows.append({
            "sequence":name,"gt_present":bool(initial_gt.any()),
            "predicted_present":bool(initial_prediction.any()),
            "dice":(float(initial_scores["dice"]) if initial_gt.any() else None),
            "iou":(float(initial_scores["iou"]) if initial_gt.any() else None),
            "empty_false_alarm":bool(not initial_gt.any() and initial_prediction.any()),
        })
        predictions=record["predictions"]
        if len(predictions) != len(frames)-source_idx-1:
            raise ValueError("Incomplete post-prompt frame coverage")
        previous=None
        items=[]
        for frame,item in zip(frames[source_idx+1:],predictions):
            expected=Path("predictions")/name/f"{frame.frame_index:06d}.png"
            if (item.get("frame_index") != frame.frame_index
                    or item.get("prediction_file") != expected.as_posix()
                    or item.get("source_image_sha256") != sha256(frame.image_path)
                    or item.get("source_gt_sha256") != sha256(frame.mask_path)):
                raise ValueError("FGPI prediction/source frame mismatch")
            path=(root/expected).resolve(strict=True)
            if not path.is_relative_to(root) or sha256(path) != item.get("prediction_sha256"):
                raise ValueError("FGPI prediction bytes/path differ")
            with Image.open(path) as image:
                prediction=np.asarray(image.convert("L")) >= 128
            gt=load_label_mask(frame.mask_path,[label_id]) == label_id
            if prediction.shape != gt.shape:
                raise ValueError("FGPI prediction geometry differs from GT")
            scores=calculate_scores(prediction,gt)
            temporal=(adjacent_mask_metrics(previous,prediction) if previous is not None
                      else undefined_temporal_metrics())
            previous=prediction
            entry=dict(sequence=name,frame=frame.frame_index,gt_present=bool(gt.any()),
                       predicted_present=bool(prediction.any()),**scores,**temporal)
            items.append(entry)
            frames_all.append(entry)
        present=[r for r in items if r["gt_present"]]
        absent=[r for r in items if not r["gt_present"]]
        sequences.append({
            "sequence":name, "frames":len(items), "present_frames":len(present),
            "empty_frames":len(absent),
            "dice":float(np.mean([r["dice"] for r in present])) if present else None,
            "iou":float(np.mean([r["iou"] for r in present])) if present else None,
            "empty_false_alarms":sum(r["predicted_present"] for r in absent),
            "initial_gt_present":bool(initial_gt.any()),
            "initial_dice":(float(initial_scores["dice"]) if initial_gt.any() else None),
            "initial_iou":(float(initial_scores["iou"]) if initial_gt.any() else None),
            "initial_false_alarm":bool(not initial_gt.any() and initial_prediction.any())
        })
    present_frames=[r for r in frames_all if r["gt_present"]]
    valid=[r for r in sequences if r["dice"] is not None]
    absent_frames=[r for r in frames_all if not r["gt_present"]]
    init_positive=[r for r in initialization_rows if r["gt_present"]]
    init_empty=[r for r in initialization_rows if not r["gt_present"]]
    summary={
        "method":"FGPI","mode":receipt["mode"],"protocol":PROTOCOL,
        "detector_export_verified":receipt["detector_export_verified"],
        "all_sequences":len(videos),"prompted_sequences":len(sequences),
        "unprompted_sequences":excluded,
        "post_prompt_frames":len(frames_all),
        "initialization_frames":len(initialization_rows),
        "initialization_gt_present_frames":len(init_positive),
        "initialization_gt_empty_frames":len(init_empty),
        "initialization_present_dice":(
            float(np.mean([r["dice"] for r in init_positive])) if init_positive else None),
        "initialization_present_iou":(
            float(np.mean([r["iou"] for r in init_positive])) if init_positive else None),
        "initialization_empty_false_alarms":sum(r["empty_false_alarm"] for r in init_empty),
        "initialization_empty_false_alarm_rate":(
            float(np.mean([r["empty_false_alarm"] for r in init_empty]))
            if init_empty else None),
        "gt_present_frames":len(present_frames),"gt_empty_frames":len(absent_frames),
        "present_frame_weighted_dice":(
            float(np.mean([r["dice"] for r in present_frames])) if present_frames else None),
        "present_equal_sequence_dice":(
            float(np.mean([r["dice"] for r in valid])) if valid else None),
        "empty_false_alarms":sum(r["predicted_present"] for r in absent_frames),
        "empty_false_alarm_rate":(
            float(np.mean([r["predicted_present"] for r in absent_frames])) if absent_frames else None),
        "warning":(
            "PolypGen is a colonoscopy proxy; no clinical adenoid efficacy inferred. "
            + ("YOLO exporter metadata was authenticated."
               if receipt["detector_export_verified"] else
               "Unverified manually supplied detections: exploratory use only.")
        ),
    }
    return summary,frames_all,sequences


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","detections","detector-checkpoint","run","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():
        p.error("Refusing to overwrite evaluation output")
    summary,frames,sequences=evaluate_fgpi_run(
        a.manifest,a.detections,a.detector_checkpoint,a.run)
    a.output.mkdir(parents=True)
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    for name,items in (("per_frame.csv",frames),("per_sequence.csv",sequences)):
        if items:
            with (a.output/name).open("w",newline="",encoding="utf-8") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(items[0]))
                writer.writeheader()
                writer.writerows(items)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
