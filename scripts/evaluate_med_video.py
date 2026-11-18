"""Evaluate outputs of infer_med_video under its exact post-oracle protocol.

No GT is used at inference after the initial prompt. This scorer validates
run provenance, disallows silent missing prediction files, and reports
foreground-present Dice separately from empty-frame false alarms.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.video import load_manifest
from datasets.io import load_label_mask
from evaluation.segmentation import calculate_scores
from evaluation.temporal_consistency import adjacent_mask_metrics, undefined_temporal_metrics
from scripts.infer_med_video import sha256

def evaluate_run(manifest, run_root):
    manifest, run_root = Path(manifest).resolve(strict=True), Path(run_root).resolve(strict=True)
    record = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    if record.get("protocol") != "oracle_first_present_mask_post_prompt_v1":
        raise ValueError("Unsupported MedSAM2 inference evaluation protocol")
    if record.get("manifest_sha256") != sha256(manifest):
        raise ValueError("Dataset manifest changed after inference")
    label_id = record["label_id"]
    videos = load_manifest(manifest, record["split"])
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
        if not (load_label_mask(frames[prompt].mask_path, [label_id]) == label_id).any():
            raise ValueError("Oracle prompt is absent")
        predictions = summary["predictions"]
        if len(predictions) != len(frames) - prompt - 1:
            raise ValueError("Missing or extra post-prompt predictions")
        sequence_scores, previous = [], None
        for frame, row in zip(frames[prompt+1:], predictions):
            if row.get("frame_index") != frame.frame_index:
                raise ValueError(f"Prediction frame order differs for {name}")
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
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Refusing to overwrite evaluation output")
    result, frame_rows, seq_rows = evaluate_run(args.manifest, args.run)
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
