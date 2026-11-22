"""Export frozen, source-hashed YOLO boxes for the FGPI research pipeline.

This is the actual YOLO inference step. It produces the exact JSONL schema
consumed by infer_fgpi_video.py, without reading annotation pixel values.
Every frame receives at most one detection for the declared target class.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from datasets.video import load_manifest
from scripts.infer_med_video import sha256


def export_detections(manifest, model, detector_checkpoint, output, *,
                      split="dev", target_class=0, yolo_conf=0.01,
                      image_size=640, iou=0.7, device="cpu",
                      image_only_manifest=False):
    manifest=Path(manifest).resolve(strict=True)
    checkpoint=Path(detector_checkpoint).resolve(strict=True)
    output=Path(output).resolve()
    metadata=output.with_name(output.name+".metadata.json")
    if output.exists() or metadata.exists():
        raise FileExistsError("Refusing to overwrite frozen YOLO detections")
    if (type(target_class) is not int or target_class < 0
            or type(image_size) is not int or image_size < 64
            or not 0 <= yolo_conf <= 1 or not 0 < iou <= 1):
        raise ValueError("Invalid YOLO class, inference image size, confidence or NMS IoU")
    if type(image_only_manifest) is not bool:
        raise ValueError("image_only_manifest must be boolean")
    videos=load_manifest(manifest,split,require_contiguous=True,
                         require_jpeg=True,safe_output_ids=True,
                         require_masks=not image_only_manifest)
    rows=[]
    detected=0
    for name,frames in sorted(videos.items()):
        for frame in frames:
            before=sha256(frame.image_path)
            result=model.predict(source=str(frame.image_path),imgsz=image_size,
                                 conf=yolo_conf,iou=iou,device=device,
                                 max_det=100,verbose=False)
            if not isinstance(result,(list,tuple)) or len(result)!=1:
                raise RuntimeError("YOLO must return exactly one result per input image")
            found=result[0].boxes
            with Image.open(frame.image_path) as image:
                width,height=image.size
            if before!=sha256(frame.image_path):
                raise RuntimeError("YOLO source image changed during inference")
            if found is None:
                continue
            xy=found.xyxy.detach().float().cpu()
            conf=found.conf.detach().float().cpu()
            classes=found.cls.detach().float().cpu()
            if xy.ndim!=2 or xy.shape[1]!=4 or conf.shape!=(len(xy),) or classes.shape!=(len(xy),):
                raise RuntimeError("Invalid YOLO box tensor geometry")
            valid=[]
            for index in range(len(xy)):
                c=float(classes[index])
                if not c.is_integer() or int(c)!=target_class:
                    continue
                box=[float(v) for v in xy[index]]
                score=float(conf[index])
                if not all(v==v and abs(v)<float("inf") for v in (*box,score)):
                    raise ValueError("Nonfinite YOLO output")
                if not 0 <= score <= 1:
                    raise ValueError("Invalid YOLO confidence")
                x0,y0,x1,y1=box
                if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                    raise ValueError("YOLO box does not fit original image geometry")
                valid.append((score,box))
            if not valid:
                continue
            # Deterministic tie-breaking for multiple same-class detections.
            score,box=min(valid,key=lambda item:(-item[0],tuple(item[1])))
            rows.append({
                "video_id":name,"frame_index":frame.frame_index,
                "box_xyxy":box,"confidence":score,"image_sha256":before,
            })
            detected+=1
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as stream:
        for record in rows:
            stream.write(json.dumps(record,sort_keys=True)+"\n")
    report={
        "schema":"adeseg_fgpi_yolo_single_box_per_frame_v1",
        "manifest_sha256":sha256(manifest),
        "detector_checkpoint_sha256":sha256(checkpoint),
        "detections_sha256":sha256(output),
        "split":split,"target_class":target_class,
        "yolo_conf":yolo_conf,"image_size":image_size,"iou":iou,
        "device":device,"detected_frames":detected,
        "total_frames":sum(len(frames) for frames in videos.values()),
        "policy":"top-confidence target-class YOLO detection per frame",
        "ground_truth_used_in_detection":False,
        "image_only_manifest": bool(image_only_manifest),
    }
    metadata.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=("dev","test"),default="dev")
    p.add_argument("--target-class",type=int,required=True,
                   help="Class ID for the target lesion in the YOLO detector")
    p.add_argument("--yolo-conf",type=float,default=0.01)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--iou",type=float,default=0.7)
    p.add_argument("--device",default="cpu")
    p.add_argument("--image-only-manifest", action="store_true",
                   help="Accept only a manifest with no GT mask fields")
    a=p.parse_args(argv)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        p.error(f"Install compatible ultralytics for YOLO export: {error}")
    model=YOLO(str(a.checkpoint.resolve(strict=True)))
    print(json.dumps(export_detections(
        a.manifest,model,a.checkpoint,a.output,split=a.split,
        target_class=a.target_class,yolo_conf=a.yolo_conf,
        image_size=a.imgsz,iou=a.iou,device=a.device,
        image_only_manifest=a.image_only_manifest),indent=2))


if __name__=="__main__":
    main()
