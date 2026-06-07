#!/usr/bin/env python3
"""Run MedSAM3 text-prompt inference on PolypGen sequences."""

import argparse
import json
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "cache"))

MEDSAM3_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MEDSAM3_ROOT.parents[1]
for path in (PROJECT_ROOT, MEDSAM3_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
from PIL import Image

from scripts.utils.eval import run_sequence_evaluation


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def natural_key(path):
    stem = Path(path).stem
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", stem)]


def discover_sequences(base_dir, seq_nums=None):
    if seq_nums:
        return [f"seq{int(seq)}" for seq in seq_nums]
    seq_dirs = [p.name for p in Path(base_dir).iterdir() if p.is_dir() and re.fullmatch(r"seq\d+", p.name)]
    return sorted(seq_dirs, key=lambda name: int(name[3:]))


def list_images(seq_dir):
    image_dir = Path(seq_dir) / "images"
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES],
        key=natural_key,
    )


def combine_prompt_masks(results, image_size, max_detections=0):
    width, height = image_size
    combined = np.zeros((height, width), dtype=bool)

    detections = []
    for key, result in results.items():
        if key == "_image" or result.get("num_detections", 0) == 0:
            continue
        masks = result.get("masks")
        scores = result.get("scores")
        if masks is None:
            continue
        for idx, mask in enumerate(masks):
            score = float(scores[idx]) if scores is not None else 0.0
            detections.append((score, np.asarray(mask).astype(bool)))

    detections.sort(key=lambda item: item[0], reverse=True)
    if max_detections > 0:
        detections = detections[:max_detections]

    for _, mask in detections:
        if mask.shape != combined.shape:
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_img = mask_img.resize((width, height), resample=Image.Resampling.NEAREST)
            mask = np.asarray(mask_img) > 127
        combined |= mask

    return combined, len(detections)


def save_mask(mask, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_path)


def write_log(log_dir, seq_name, payload):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{seq_name}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return log_path


def read_log(log_dir, seq_name):
    log_path = log_dir / f"{seq_name}.json"
    if not log_path.exists():
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_sequence(inferencer, args, seq_name):
    seq_dir = Path(args.base_dir) / seq_name
    image_paths = list_images(seq_dir)
    pred_dir = Path(args.output_root) / args.method_name / "mask" / seq_name / "predicted"
    log_dir = Path(args.output_root) / args.method_name / "logs"
    previous_log = read_log(log_dir, seq_name) if args.resume_failed else None

    if previous_log:
        failed_frames = {
            frame["frame"]
            for frame in previous_log.get("frames", [])
            if frame.get("status") != "ok"
        }
        image_paths = [path for path in image_paths if path.name in failed_frames]
        if not image_paths:
            print(f"  No failed frames to resume for {seq_name}; skipping.")
            return

    log = {
        "sequence": seq_name,
        "prompt": args.prompt,
        "threshold": args.threshold,
        "nms_iou": args.nms_iou,
        "resolution": args.resolution,
        "device": str(inferencer.device),
        "max_detections_per_frame": args.max_detections_per_frame,
        "num_frames": len(image_paths),
        "frames": [],
        "status": "running",
    }
    if previous_log:
        log["resume_failed"] = True
        log["num_total_frames"] = previous_log.get("num_total_frames", previous_log.get("num_frames"))

    if not image_paths:
        log["status"] = "failed"
        log["error"] = f"No images found in {seq_dir / 'images'}"
        write_log(log_dir, seq_name, log)
        return

    for image_path in image_paths:
        frame_log = {"frame": image_path.name}
        try:
            results = inferencer.predict(str(image_path), args.prompt)
            image = results["_image"]
            mask, num_detections = combine_prompt_masks(
                results,
                image.size,
                max_detections=args.max_detections_per_frame,
            )
            output_path = pred_dir / f"{image_path.stem}.png"
            save_mask(mask, output_path)
            frame_log.update({"status": "ok", "num_detections": num_detections})
        except Exception as exc:
            frame_log.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if args.blank_on_frame_error:
                with Image.open(image_path) as image:
                    blank = np.zeros((image.height, image.width), dtype=bool)
                output_path = pred_dir / f"{image_path.stem}.png"
                save_mask(blank, output_path)
        log["frames"].append(frame_log)

    if previous_log:
        merged_frames = {frame["frame"]: frame for frame in previous_log.get("frames", [])}
        merged_frames.update({frame["frame"]: frame for frame in log["frames"]})
        log["frames"] = list(merged_frames.values())
        log["num_frames"] = previous_log.get("num_frames", len(log["frames"]))
        log["num_resumed_frames"] = len(image_paths)

    failures = [frame for frame in log["frames"] if frame["status"] != "ok"]
    log["status"] = "completed" if not failures else "completed_with_frame_failures"
    log["num_failed_frames"] = len(failures)
    log["num_saved_masks"] = len(list(pred_dir.glob("*.png"))) if pred_dir.exists() else 0
    write_log(log_dir, seq_name, log)


def main():
    parser = argparse.ArgumentParser(description="Run MedSAM3 text-prompt inference on PolypGen.")
    parser.add_argument("--base_dir", default="data/polypgen", help="PolypGen root with seq*/images folders.")
    parser.add_argument("--seq_nums", nargs="+", type=int, help="Sequence numbers to process. Omit for all.")
    parser.add_argument("--output_root", default="outputs", help="Root output directory.")
    parser.add_argument("--method_name", default="MedSAM3_TEXT_POLYP", help="Output method name.")
    parser.add_argument("--bbox_root", default="data/bbox", help="Ground-truth bbox root for evaluation.")
    parser.add_argument("--config", default=str(MEDSAM3_ROOT / "configs" / "full_lora_config.yaml"))
    parser.add_argument("--weights", default=None, help="MedSAM3 LoRA weights. If omitted, infer_sam.py uses config output_dir.")
    parser.add_argument("--base_checkpoint", default=None, help="Optional local facebook/sam3 base checkpoint path.")
    parser.add_argument("--prompt", nargs="+", default=["polyp"], help="Text prompt(s), for example: polyp.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--nms_iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--resolution", type=int, default=1008, help="Square inference resolution.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Inference device. auto prefers CUDA, then CPU. MPS is opt-in because SAM3 geometry encoding is unstable on it.",
    )
    parser.add_argument(
        "--max_detections_per_frame",
        type=int,
        default=0,
        help="Keep top-k detections per frame after NMS. 0 means keep all.",
    )
    parser.add_argument(
        "--blank_on_frame_error",
        action="store_true",
        help="Write a blank mask if one frame fails, so evaluation can still include it.",
    )
    parser.add_argument(
        "--resume_failed",
        action="store_true",
        help="Read existing logs and rerun only frames whose previous status was not ok.",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Run mask evaluation after inference.",
    )
    args = parser.parse_args()

    sequences = discover_sequences(args.base_dir, args.seq_nums)
    print(f"Processing {len(sequences)} sequences: {sequences}")
    print(f"Text prompt(s): {args.prompt}")

    try:
        from infer_sam import SAM3LoRAInference
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import MedSAM3 inference dependencies. "
            f"Missing module: {exc.name}. Install the MedSAM3/SAM3 environment first."
        ) from exc

    inferencer = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        base_checkpoint_path=args.base_checkpoint,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou,
        device=args.device,
    )

    for seq_name in sequences:
        print(f"\nProcessing {seq_name}...")
        infer_sequence(inferencer, args, seq_name)

    if args.run_eval:
        run_sequence_evaluation(
            sequence_names=sequences,
            data_root=args.base_dir,
            bbox_root=args.bbox_root,
            output_root=args.output_root,
            method_name=args.method_name,
            prompt_source="text",
            verbose=True,
        )


if __name__ == "__main__":
    main()
