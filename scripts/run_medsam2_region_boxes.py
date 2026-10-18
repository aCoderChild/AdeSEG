#!/usr/bin/env python3
"""YOLO-box comparison adapted from d7f7192:MedSAM2/medsam2_infer_frame.py.

Model/decoder/memory are the bundled author implementation. Region bias and
tracking reuse this project's existing hooks. This is a separate prospective
MedSAM2-Tiny experiment, not a result of the registered SAM2-B+ study.
"""

# ruff: noqa: E402 -- direct CLI entry point

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from evaluation.segmentation import calculate_scores
from modeling.medsam2 import build_video_predictor
from modeling.region_tokens import reliability_bias, token_categories
from modeling.memory_attention import UtilityController, attach_utility_controller
from scripts.run_region_token_pilot import write_json
from training.tracking import start_wandb, log_results_artifact


def box_points(box, height, width, image_size, device):
    """SAM2's native box-as-two-corners convention (labels 2 and 3)."""
    coords = torch.as_tensor(box, dtype=torch.float32, device=device).reshape(1, 2, 2)
    coords = coords / coords.new_tensor([width, height]) * image_size
    return {"point_coords": coords, "point_labels": torch.tensor([[2, 3]], device=device, dtype=torch.int32)}


def detect(yolo, frame, device):
    # Match the recovered frame baseline: RGB PIL input, confidence .5,
    # image size 640, highest-confidence detection only, no GT fallback.
    with Image.open(frame.image_path) as image:
        result = yolo.predict([image.convert("RGB")], imgsz=640, conf=0.5,
                              device=device, verbose=False)[0]
    if result.boxes is None or not len(result.boxes):
        return None
    top = int(np.argsort(result.boxes.conf.cpu().numpy())[::-1][0])
    return {"box": result.boxes.xyxy[top].cpu().tolist(),
            "confidence": float(result.boxes.conf[top])}


def bias_for_bank(units, predictions, frames, grid, weights, scale, arm, seed):
    taxonomy = {}
    for unit in units:
        t = unit["frame_idx"]
        image = cv2.cvtColor(cv2.imread(str(frames[t].image_path)), cv2.COLOR_BGR2RGB)
        taxonomy[t] = token_categories(predictions[t], grid=(grid, grid), image=image)
    if arm == "reliability_only":
        weights = [1., 1., 1., 1.]
    return reliability_bias(units, taxonomy, weights, scale,
                            "no_reliability" if arm == "region_only" else "full", seed)


def check_development_gate(folder, checkpoint, device):
    """Require completed independent zero-scale rollouts and a close replay."""
    folder = Path(folder)
    status = json.loads((folder / "status.json").read_text())
    setup = json.loads((folder / "setup.json").read_text())
    if status.get("state") != "complete" or setup.get("scale") != 0:
        raise ValueError("Completed zero-scale validation required")
    if setup.get("device") != device or setup.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError("Validation device/checkpoint mismatch")
    rows = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines() if line]
    replay = [r for r in rows if r["arm"] == "framewise"]
    if not replay or max(abs(r["dice"] - r["historical_dice"]) for r in replay) > 1e-4:
        raise ValueError("Historical per-frame Dice replay differs by more than 0.0001")
    if not any(v["memory_calls"]["memory"] > 0 for v in json.loads((folder / "summary.json").read_text()).values()):
        raise ValueError("Validation never exercised memory attention")


@torch.inference_mode()
def run_video(predictor, controller, frames, detections, output, arms, weights, scale, reference=None, save_masks=True):
    from external.MedSAM2.sam2.sam2_image_predictor import SAM2ImagePredictor

    image_predictor = SAM2ImagePredictor(predictor)
    rows, hashes, memory_calls = [], {}, {}
    grid = predictor.image_size // predictor.backbone_stride
    with tempfile.TemporaryDirectory(prefix="frames_", dir=output) as directory:
        stage_frames(frames, Path(directory))
        for arm in arms:
            predictions, initialized = {}, False
            controller.token_bias_policy = None
            controller.token_biases = {}
            calls = [0]
            handle = predictor.memory_attention.register_forward_hook(
                lambda *unused: calls.__setitem__(0, calls[0] + 1))
            state = predictor.init_state(video_path=directory, offload_video_to_cpu=True,
                                         offload_state_to_cpu=True)
            bank = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            if arm not in {"framewise", "memory", "video_no_memory", "detector_refresh"}:
                controller.token_bias_policy = lambda units: bias_for_bank(
                    units, predictions, frames, grid, weights,
                    0. if arm == "zero_bias" else scale, arm, 0)
            arm_hashes = []
            try:
                for t, frame in enumerate(frames):
                    detection = detections[t]
                    object_score, predicted_iou = None, None
                    with Image.open(frame.image_path) as source:
                        image = source.convert("RGB")
                        width, height = image.size
                        if arm == "framewise":
                            if detection is None:
                                mask = np.zeros((height, width), dtype=bool)
                            else:
                                image_predictor.set_image(image)
                                mask = image_predictor.predict(
                                    box=np.asarray(detection["box"], np.float32),
                                    multimask_output=False)[0][0].astype(bool)
                        elif detection is None and (not initialized or arm == "video_no_memory"):
                            mask = np.zeros((height, width), dtype=bool)
                        else:
                            points = None if detection is None else box_points(
                                detection["box"], height, width, predictor.image_size, predictor.device)
                            current, logits = predictor.run_single_frame_inference(
                                inference_state=state, output_dict=bank, frame_idx=t,
                                batch_size=1, is_init_cond_frame=not initialized or arm == "video_no_memory"
                                or (arm == "detector_refresh" and detection is not None),
                                point_inputs=points, mask_inputs=None, reverse=False,
                                run_mem_encoder=arm != "video_no_memory")
                            if arm != "video_no_memory":
                                bank["non_cond_frame_outputs" if initialized else "cond_frame_outputs"][t] = current
                            object_score = float(current["object_score_logits"].float().mean().cpu())
                            predicted_iou = float(current["iou_predictions"].float().mean().cpu())
                            initialized = True
                            mask = (F.interpolate(logits.float(), (height, width),
                                    mode="bilinear", align_corners=False)[0, 0].cpu().numpy() > 0)
                            predictions[t] = mask
                    arm_hashes.append(hashlib.sha256(mask.tobytes()).hexdigest())
                    if reference is not None and arm == "framewise":
                        expected = reference[frame.video_id][t]
                        if hashlib.sha256(np.asarray(image).tobytes()).hexdigest() != expected["image_sha256"]:
                            raise RuntimeError("Historical reference and current source image differ")
                    if save_masks:
                        destination = output / "masks" / arm / frame.video_id / f"{frame.image_path.stem}.png"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(mask.astype(np.uint8) * 255).save(destination)
                    # Labels are accessed only after the prediction is finalized.
                    truth = load_label_mask(frame.mask_path, [1]) > 0
                    row = {"video": frame.video_id, "frame": t, "arm": arm,
                           "detected": detection is not None, "present": bool(truth.any()),
                           "detector_refresh": arm == "detector_refresh" and detection is not None,
                           "object_score_logit": object_score, "predicted_iou": predicted_iou,
                           "prediction_area": int(mask.sum()), "ground_truth_area": int(truth.sum()),
                           **calculate_scores(mask, truth)}
                    if reference is not None and arm == "framewise":
                        row.update(historical_mask_exact=arm_hashes[-1] == expected["mask_sha256"],
                                   historical_dice=expected["dice"])
                    rows.append(row)
                    with (output / "frames.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    print(f"[box-region] {frame.video_id} {arm} frame={t+1}/{len(frames)}", flush=True)
            finally:
                handle.remove()
                controller.token_bias_policy = None
                controller.token_biases = {}
                predictor.reset_state(state)
            hashes[arm], memory_calls[arm] = arm_hashes, calls[0]
    if "zero_bias" in hashes and hashes["zero_bias"] != hashes["memory"]:
        raise RuntimeError("Independent zero-bias rollout differs from ordinary memory")
    if scale == 0 and "memory" in hashes and any(hashes[arm] != hashes["memory"] for arm in arms if arm not in {"framewise", "video_no_memory", "detector_refresh"}):
        raise RuntimeError("A zero-scale method arm differs from ordinary memory")
    for arm in arms:
        if arm == "video_no_memory" and memory_calls[arm]:
            raise RuntimeError("Memory-disabled video control unexpectedly read memory")
        if arm == "detector_refresh":
            first = next((i for i, d in enumerate(detections) if d is not None), len(detections))
            if memory_calls[arm] != sum(d is None for d in detections[first + 1:]):
                raise RuntimeError("Detector refresh must read memory exactly on missing-detection frames")
        elif arm not in {"framewise", "video_no_memory"} and sum(d is not None for d in detections) >= 2 and not memory_calls[arm]:
            raise RuntimeError(f"No memory attention executed for {arm}")
    return rows, memory_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--yolo-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--selection", type=Path,
                        help="Frozen development-only selection; required for held-out evaluation")
    parser.add_argument("--gate-dir", type=Path)
    parser.add_argument("--arm", action="append", choices=["framewise", "video_no_memory", "memory", "zero_bias", "reliability_only", "region_only", "full", "detector_refresh"])
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--detections-dir", type=Path,
                        help="Reuse frozen per-frame detections from an earlier matched run")
    parser.add_argument("--max-frames", type=int, default=12,
                        help="Index-only technical validation prefix, not full-sequence results")
    parser.add_argument("--reference", type=Path,
                        help="Historical source-image/mask hashes and per-frame Dice")
    parser.add_argument("--weights", type=float, nargs=4, default=[1., 1., 1., 1.])
    parser.add_argument("--scale", type=float, default=0.)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not np.isfinite(args.weights).all() or min(args.weights) <= 0 or not np.isfinite(args.scale) or args.scale < 0:
        parser.error("Weights must be finite positive numbers; scale must be finite and nonnegative")
    # No test-time hyperparameter search.
    if sum((args.validation, args.development, args.evaluation)) != 1:
        parser.error("Choose exactly one of --validation, --development or --evaluation")
    if args.validation and (args.max_frames < 2 or args.reference is None):
        parser.error("Validation requires a historical reference and at least two frames")
    if args.development or args.evaluation:
        permitted = range(1, 16) if args.development else range(16, 24)
        if not args.gate_dir or not set(args.sequence) <= {f"seq{i}" for i in permitted}:
            parser.error("Development/evaluation IDs must follow protocol_a and require --gate-dir")
        check_development_gate(args.gate_dir, args.checkpoint, args.device)
    if args.evaluation:
        if args.selection is None or args.max_frames != 0:
            parser.error("Held-out evaluation requires a frozen --selection and --max-frames 0")
        selection = json.loads(args.selection.read_text())
        if selection.get("checkpoint_sha256") != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
            parser.error("Frozen selection checkpoint differs")
        if selection.get("development_sequences") != [f"seq{i}" for i in range(1, 16)]:
            parser.error("Frozen selection must include every development sequence")
        args.weights, args.scale = selection["weights"], selection["scale"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.reference.read_text()) if args.reference else None
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config.update(source="d7f7192:MedSAM2/medsam2_infer_frame.py", git_revision=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        config="configs/sam2.1_hiera_t512.yaml", prompt_policy="same per-frame detector boxes; memory propagates missing detections after initialization",
        historical_baseline_reproduced=False, full_study_complete=False)
    for name, path in (("checkpoint", args.checkpoint), ("yolo_checkpoint", args.yolo_checkpoint)):
        config[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(args.output_dir / "setup.json", config)
    phase = "validation" if args.validation else "development" if args.development else "evaluation"
    run = start_wandb("ade-seg", "adenoid-hypertrophy", "MedSAM2 region boxes | " + phase,
                      "online", "https://forge.coreweave.com/api/wandb", config,
                      args.output_dir, ["region-boxes", phase])
    try:
        from ultralytics import YOLO
        yolo = YOLO(str(args.yolo_checkpoint))
        predictor = build_video_predictor(config["config"], args.checkpoint, args.device,
                                           apply_postprocessing=False).eval()
        controller = UtilityController()
        controller.enabled = False
        attach_utility_controller(predictor, controller)
        videos = load_manifest(args.manifest, "dev" if args.development else "test")
        summary = {}
        for video in args.sequence:
            frames = videos[video][:args.max_frames] if args.max_frames > 0 else videos[video]
            if args.detections_dir:
                detections = json.loads((args.detections_dir / f"detections_{video}.json").read_text())
                if len(detections) != len(frames):
                    raise ValueError("Frozen detection count differs from frame count")
            else:
                detections = [detect(yolo, frame, args.device) for frame in frames]
            write_json(args.output_dir / f"detections_{video}.json", detections)
            rows, calls = run_video(predictor, controller, frames, detections, args.output_dir,
                args.arm or ["framewise", "memory", "zero_bias", "reliability_only", "region_only", "full"],
                args.weights, args.scale, reference, args.save_masks or args.validation)
            summary[video] = {arm: float(np.mean([r["dice"] for r in rows if r["arm"] == arm]))
                              for arm in {r["arm"] for r in rows}}
            summary[video]["memory_calls"] = calls
            replay = [r for r in rows if r["arm"] == "framewise"]
            if reference is not None:
                summary[video]["historical_masks_exact"] = sum(r["historical_mask_exact"] for r in replay)
                summary[video]["historical_frames"] = len(replay)
                summary[video]["historical_mean_dice_delta"] = float(np.mean([
                    r["dice"] - r["historical_dice"] for r in replay]))
            write_json(args.output_dir / "summary.json", summary)
            run.log({f"{video}/{key}": value for key, value in summary[video].items() if key != "memory_calls"})
        write_json(args.output_dir / "summary.json", summary)
        write_json(args.output_dir / "status.json", {"state": "complete", "validation_only": args.validation,
                                                       "historical_baseline_reproduced": False})
        names = [str(p.relative_to(args.output_dir)) for p in args.output_dir.rglob("*")
                 if p.is_file() and "wandb" not in p.relative_to(args.output_dir).parts]
        log_results_artifact(run, args.output_dir, names)
    except Exception:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
