"""Evaluate outputs of infer_med_video under its exact post-oracle protocol.

No GT is used at inference after the initial prompt. This scorer validates
run provenance, disallows silent missing prediction files, and reports
foreground-present Dice separately from empty-frame false alarms.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.video import load_manifest
from datasets.io import load_label_mask
from evaluation.segmentation import calculate_scores
from evaluation.temporal_consistency import adjacent_mask_metrics, undefined_temporal_metrics
from scripts.infer_med_video import sha256
from adeseg.research.med_video_contract import (
    load_med_video_contract, verify_med_run_receipt,
)

def evaluate_run(manifest, run_root, *, protocol=None):
    manifest, run_root = Path(manifest).resolve(strict=True), Path(run_root).resolve(strict=True)
    record = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    if protocol is not None:
        contract = load_med_video_contract(protocol)
        if manifest != contract["manifest"]:
            raise ValueError("Evaluation manifest differs from sealed protocol")
        verify_med_run_receipt(contract, record)
    if record.get("protocol") != "oracle_first_present_mask_post_prompt_v1":
        raise ValueError("Unsupported MedSAM2 inference evaluation protocol")
    if record.get("manifest_sha256") != sha256(manifest):
        raise ValueError("Dataset manifest changed after inference")
    label_id = record["label_id"]
    videos = load_manifest(manifest, record["split"], require_contiguous=True,
                           require_jpeg=True, safe_output_ids=True)
    if videos.keys() != record["results"].keys():
        raise ValueError("Run video population differs from the dataset manifest")
    results, frame_rows, sequence_rows = record["results"], [], []
    for name, frames in sorted(videos.items()):
        summary = results[name]
        if summary["status"] == "unpromptable":
            if any((load_label_mask(f.mask_path, [label_id]) == label_id).any()
                   for f in frames):
                raise ValueError(f"Unpromptable video contains foreground: {name}")
            continue
        if summary["status"] != "completed":
            raise ValueError(f"Unsuccessful inference for {name}")
        prompt = summary["prompt_position"]
        if not isinstance(prompt, int) or prompt < 0 or prompt >= len(frames):
            raise ValueError("Invalid oracle prompt position")
        if summary.get("prompt_source_index") != frames[prompt].frame_index:
            raise ValueError("Prompt source index changed")
        if summary.get("prompt_source") != "oracle_gt_mask":
            raise ValueError("Run does not declare oracle GT initialization")
        if (sha256(frames[prompt].mask_path) != summary.get("prompt_mask_sha256")
                or sha256(frames[prompt].image_path) != summary.get("prompt_image_sha256")):
            raise ValueError("Oracle prompt image/annotation changed since inference")
        if not (load_label_mask(frames[prompt].mask_path, [label_id]) == label_id).any():
            raise ValueError("Oracle prompt is absent")
        if any((load_label_mask(frame.mask_path, [label_id]) == label_id).any()
               for frame in frames[:prompt]):
            raise ValueError("Oracle prompt is not the first foreground frame")
        predictions = summary["predictions"]
        if len(predictions) != len(frames) - prompt - 1:
            raise ValueError("Missing or extra post-prompt predictions")
        sequence_scores, previous = [], None
        for frame, row in zip(frames[prompt+1:], predictions):
            if row.get("frame_index") != frame.frame_index:
                raise ValueError(f"Prediction frame order differs for {name}")
            if (sha256(frame.image_path) != row.get("source_image_sha256")
                    or sha256(frame.mask_path) != row.get("source_gt_sha256")):
                raise ValueError(f"Source image/annotation bytes changed for {name}")
            relpath = Path(row["prediction_file"])
            expected_relpath = Path("predictions") / name / (frame.image_path.stem + ".png")
            if relpath != expected_relpath:
                raise ValueError("Prediction path does not match manifest frame")
            path = (run_root / relpath).resolve()
            if not path.is_relative_to(run_root) or not path.is_file():
                raise FileNotFoundError(path)
            if sha256(path) != row.get("prediction_sha256"):
                raise ValueError(f"Prediction SHA256 mismatch: {path}")
            prediction = np.asarray(Image.open(path).convert("L")) >= 128
            if record.get("raw_logits_recorded") and "logit_file" not in row:
                raise ValueError("Run declares float32 logits, but a frame is missing them")
            if "logit_file" in row:
                logit_relpath = Path(row["logit_file"])
                expected_logit = Path("logits") / name / (frame.image_path.stem + ".npy")
                if logit_relpath != expected_logit:
                    raise ValueError("Logit file path differs from sealed frame ordering")
                logit_path = (run_root / logit_relpath).resolve()
                if not logit_path.is_relative_to(run_root) or not logit_path.is_file():
                    raise FileNotFoundError(logit_path)
                if sha256(logit_path) != row.get("logit_file_sha256"):
                    raise ValueError("Saved float32 logit file SHA256 mismatch")
                logits = np.load(logit_path, allow_pickle=False)
                if (logits.dtype != np.float32 or logits.shape != prediction.shape
                        or not np.isfinite(logits).all()):
                    raise ValueError("Saved float32 logits have incompatible shape/dtype")
                actual_hash = hashlib.sha256(
                    np.ascontiguousarray(logits).tobytes()
                ).hexdigest()
                if actual_hash != row.get("logit_sha256"):
                    raise ValueError("Saved float32 logits differ from recorded tensor hash")
                if not np.array_equal(logits > 0, prediction):
                    raise ValueError("Logits and exported segmentation PNG disagree")
            ground_truth = load_label_mask(frame.mask_path, [label_id]) == label_id
            if prediction.shape != ground_truth.shape:
                raise ValueError(f"Prediction/GT shape mismatch: {name}:{frame.frame_index}")
            scores = calculate_scores(prediction, ground_truth)
            temporal = (adjacent_mask_metrics(previous, prediction)
                        if previous is not None else undefined_temporal_metrics())
            previous = prediction
            item = {
                "sequence": name, "frame": frame.frame_index,
                "gt_present": bool(ground_truth.any()),
                "predicted_present": bool(prediction.any()),
                **scores, **temporal,
            }
            frame_rows.append(item)
            sequence_scores.append(item)
        present = [x for x in sequence_scores if x["gt_present"]]
        absent = [x for x in sequence_scores if not x["gt_present"]]
        sequence_rows.append({
            "sequence": name, "frames": len(sequence_scores),
            "present_frames": len(present), "empty_frames": len(absent),
            "dice": float(np.mean([x["dice"] for x in present])) if present else float("nan"),
            "iou": float(np.mean([x["iou"] for x in present])) if present else float("nan"),
            "empty_false_alarms": sum(x["predicted_present"] for x in absent),
        })
    present_frames = [x for x in frame_rows if x["gt_present"]]
    absent_frames = [x for x in frame_rows if not x["gt_present"]]
    valid_sequences = [x for x in sequence_rows if np.isfinite(x["dice"])]
    result = {
        "protocol": record["protocol"],
        "all_post_prompt_frames": len(frame_rows),
        "gt_present_frames": len(present_frames),
        "gt_empty_frames": len(absent_frames),
        "unpromptable_sequences": sorted(k for k,v in results.items()
                                        if v["status"] == "unpromptable"),
        "present_frame_weighted_dice": (
            float(np.mean([x["dice"] for x in present_frames])) if present_frames else None),
        "present_equal_sequence_dice": (
            float(np.mean([x["dice"] for x in valid_sequences])) if valid_sequences else None),
        "empty_false_alarms": sum(x["predicted_present"] for x in absent_frames),
        "empty_false_alarm_rate": (
            float(np.mean([x["predicted_present"] for x in absent_frames]))
            if absent_frames else None),
        "temporal_iou_mean": (
            float(np.mean([x["temporal_iou"] for x in frame_rows
                           if np.isfinite(x["temporal_iou"])]))
            if any(np.isfinite(x["temporal_iou"]) for x in frame_rows) else None),
    }
    return result, frame_rows, sequence_rows

def write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Refusing to overwrite evaluation output")
    contract = load_med_video_contract(args.protocol)
    result, frame_rows, seq_rows = evaluate_run(
        contract["manifest"], args.run, protocol=args.protocol)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    write_csv(args.output / "per_sequence.csv", seq_rows,
              ["sequence", "frames", "present_frames", "empty_frames",
               "dice", "iou", "empty_false_alarms"])
    if frame_rows:
        write_csv(args.output / "per_frame.csv", frame_rows, list(frame_rows[0]))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
