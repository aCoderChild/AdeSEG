"""Matched detector-only versus FGPI point-augmentation comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from scripts.infer_fgpi_video import PROTOCOL
from scripts.evaluate_fgpi_video import evaluate_fgpi_run


def compare_fgpi(manifest, detections, detector_checkpoint, box_run, fgpi_run):
    box_root=Path(box_run).resolve(strict=True)
    fgpi_root=Path(fgpi_run).resolve(strict=True)
    if box_root==fgpi_root:
        raise ValueError("Control and candidate must be independent runs")
    left=json.loads((box_root/"run.json").read_text())
    right=json.loads((fgpi_root/"run.json").read_text())
    if left.get("mode")!="box_only" or right.get("mode")!="fgpi":
        raise ValueError("Expected box_only control and fgpi candidate")
    fixed=("protocol","split","label_id","manifest_sha256","detections_sha256",
           "detector_checkpoint_sha256","backbone_sha256","model_config",
           "confidence_gate","postprocessing","prompt_budget",
           "point_budget","prompt_selector_source_sha256",
           "detector_export_verified","detector_export_metadata_sha256")
    if any(left.get(key)!=right.get(key) for key in fixed) or left.get("protocol")!=PROTOCOL:
        raise ValueError("Different detector/backbone/evaluation settings")
    if left.get("results",{}).keys()!=right.get("results",{}).keys():
        raise ValueError("FGPI runs differ in sequence population")
    for video in left["results"]:
        a,b=left["results"][video],right["results"][video]
        for key in ("status","frames","predicted","prompt_position",
                    "prompt_frame_index","prompt_image_sha256"):
            if a.get(key)!=b.get(key):
                raise ValueError("FGPI compared unmatched first detector prompts")
        if a.get("status")=="completed":
            if a["prompt"]["box_xyxy"]!=b["prompt"]["box_xyxy"]:
                raise ValueError("FGPI used a different detector box")
            if [r["frame_index"] for r in a["predictions"]] != [
                    r["frame_index"] for r in b["predictions"]]:
                raise ValueError("FGPI frame coverage differs")
    base,_,base_sequences=evaluate_fgpi_run(
        manifest,detections,detector_checkpoint,box_root)
    candidate,_,candidate_sequences=evaluate_fgpi_run(
        manifest,detections,detector_checkpoint,fgpi_root)
    if [r["sequence"] for r in base_sequences]!=[r["sequence"] for r in candidate_sequences]:
        raise ValueError("FGPI evaluated different video populations")
    rows=[]
    for a,b in zip(base_sequences,candidate_sequences):
        if (a["present_frames"]!=b["present_frames"]
                or a["empty_frames"]!=b["empty_frames"]):
            raise ValueError("FGPI evaluated different GT frame populations")
        rows.append({"sequence":a["sequence"],
                     "box_initial_dice":a["initial_dice"],
                     "fgpi_initial_dice":b["initial_dice"],
                     "initial_delta":(
                         b["initial_dice"]-a["initial_dice"]
                         if a["initial_dice"] is not None and b["initial_dice"] is not None
                         else None),
                     "box_initial_false_alarm":a["initial_false_alarm"],
                     "fgpi_initial_false_alarm":b["initial_false_alarm"],
                     "box_dice":a["dice"],
                     "fgpi_dice":b["dice"],
                     "delta":b["dice"]-a["dice"] if a["dice"] is not None and b["dice"] is not None else None,
                     "box_empty_false_alarms":a["empty_false_alarms"],
                     "fgpi_empty_false_alarms":b["empty_false_alarms"]})
    initial_valid=[r["initial_delta"] for r in rows if r["initial_delta"] is not None]
    valid=[r["delta"] for r in rows if r["delta"] is not None]
    ci=None
    if valid:
        rng=np.random.default_rng(20261120)
        values=np.asarray(valid,dtype=np.float64)
        samples=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
        ci=[float(x) for x in np.quantile(samples,[.025,.975])]
    return {
        "protocol":PROTOCOL,"control":base,"candidate":candidate,
        "per_sequence":rows, "eligible_sequence_count":len(valid),
        "mean_paired_dice_delta":float(np.mean(valid)) if valid else None,
        "mean_paired_initialization_dice_delta":(
            float(np.mean(initial_valid)) if initial_valid else None),
        "eligible_initialization_sequences":len(initial_valid),
        "exploratory_95ci_sequence_bootstrap":ci,
        "warning":"Sequence independence unverified; no benefit is established without held-out full-cohort evaluation",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","detections","detector-checkpoint","box-run","fgpi-run","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():
        p.error("Refusing to overwrite paired analysis")
    result=compare_fgpi(
        a.manifest,a.detections,a.detector_checkpoint,a.box_run,a.fgpi_run)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
