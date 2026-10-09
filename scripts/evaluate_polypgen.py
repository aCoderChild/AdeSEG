"""Evaluate fixed PolypGen video-propagation methods and write auditable JSON results.

Each method receives the tight first-positive-frame ground-truth box, so this evaluates
post-prompt propagation rather than fully automatic detection.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path[:0] = [str(ROOT), str(ROOT / "external" / "MedSAM2")]

from adenoid.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from modeling.detector import PolypDetector  # noqa: E402
from modeling.kalman_memory import RKNMemoryUpdate  # noqa: E402
from modeling.medsam2 import build_video_predictor  # noqa: E402
from modeling.rde_memory import RDEMemoryUpdate  # noqa: E402
from modeling.recurrent_memory import REDETECT_CONFIDENCE, load_memory_update  # noqa: E402
from infer import first_box, predict_label, stage_frames  # noqa: E402
from training.tracking import finish, log, start_wandb  # noqa: E402
from training.protocol import evaluation_role, load_protocol  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--videos", default="all", help="Comma-separated IDs, or all.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="Directory containing full/ and presence_only/ checkpoints.")
    parser.add_argument("--full-run", default="full_seed0_500steps",
                        help="Subdirectory of checkpoint-root containing the full RKN checkpoint.")
    parser.add_argument("--presence-only-run", default="presence_only_seed0_500steps",
                        help="Subdirectory of checkpoint-root containing the presence-only checkpoint.")
    parser.add_argument("--rde-run", default=None,
                        help="Subdirectory of checkpoint-root containing a trained RDE checkpoint.")
    parser.add_argument("--fixed-gain", type=float, default=None,
                        help="EMA control gain in [0, 1] for kalman_fixed_gain; select it on validation only.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl")
    parser.add_argument("--protocol", type=Path, default=None,
                        help="Optional JSON protocol; labels the run as validation, exploratory or locked_test.")
    parser.add_argument("--sam2-cfg", default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=ROOT / "checkpoints/MedSAM2_latest.pt")
    parser.add_argument("--detector", type=Path, default=ROOT / "checkpoints/polypgen_yolov8n.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--wandb-project", default="ade-seg")
    parser.add_argument("--wandb-entity", default="adenoid-hypertrophy")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-base-url", default="https://api.wandb.ai")
    parser.add_argument("--methods", nargs="+", required=True,
                        choices=("native", "native_redetect", "detector_only", "rde", "rde_redetect",
                                 "rde_trained", "rde_trained_redetect", "kalman_untrained", "presence_only",
                                 "kalman_full", "kalman_fixed_gain"))
    args = parser.parse_args()
    if "kalman_fixed_gain" in args.methods and (args.fixed_gain is None or not 0 <= args.fixed_gain <= 1):
        parser.error("kalman_fixed_gain requires --fixed-gain in [0, 1].")
    if any(method.startswith("rde_trained") for method in args.methods) and not args.rde_run:
        parser.error("rde_trained methods require --rde-run.")
    return args


def predictor_for(method, args, detector):
    if method == "native":
        return build_video_predictor(args.sam2_cfg, args.sam2_checkpoint, args.device), False
    predictor = build_video_predictor(
        args.sam2_cfg, args.sam2_checkpoint, args.device,
        predictor_target="modeling.recurrent_memory.RecurrentMemoryVideoPredictor",
    )
    predictor.detector = detector
    torch.manual_seed(0)
    if method == "native_redetect":
        predictor.redetect_enabled = True
        return predictor, False
    if method in ("rde", "rde_redetect", "rde_trained", "rde_trained_redetect"):
        if method.startswith("rde_trained"):
            predictor.memory_update = load_memory_update(
                args.checkpoint_root / args.rde_run / "memory_update.pt", predictor, predictor.device,
            )
        else:
            predictor.memory_update = RDEMemoryUpdate(predictor.mem_dim, predictor.hidden_dim).to(args.device).eval()
        predictor.redetect_enabled = method.endswith("redetect")
    elif method == "kalman_untrained":
        predictor.memory_update = RKNMemoryUpdate(predictor.mem_dim, predictor.hidden_dim, weight_factor=1.0).to(args.device).eval()
    else:
        arm = args.full_run if method in ("kalman_full", "kalman_fixed_gain") else args.presence_only_run
        predictor.memory_update = load_memory_update(
            args.checkpoint_root / arm / "memory_update.pt", predictor, predictor.device,
        )
        if method == "kalman_fixed_gain":
            predictor.memory_update.fixed_gain = args.fixed_gain
    return predictor, True


@torch.inference_mode()
def detector_only(predictor, frames, staged, detector):
    prompt, _ = first_box(frames, 1, [1], False)
    state = predictor.init_state(video_path=str(staged), offload_video_to_cpu=True)
    detections = detector([frame.image_path for frame in frames])
    height, width = state["video_height"], state["video_width"]
    logits = np.full((len(frames), height, width), -np.inf, dtype=np.float32)
    traces = {}
    for index in range(prompt + 1, len(frames)):
        box, confidence = detections[index]
        traces[index] = {"redetected": confidence >= REDETECT_CONFIDENCE}
        if confidence < REDETECT_CONFIDENCE:
            continue
        _, _, features, _, sizes = predictor._get_image_feature(state, index, 1)
        high_res = predictor._redetect(features, sizes, torch.tensor([box], device=predictor.device))[4]
        logits[index] = F.interpolate(high_res.float(), size=(height, width), mode="bilinear")[0, 0].cpu().numpy()
    predictor.reset_state(state)
    return logits, prompt, traces


def evaluate(method, videos, args, detector):
    predictor, recurrent = predictor_for("native_redetect" if method == "detector_only" else method, args, detector)
    rows = []
    for video, frames in videos.items():
        with tempfile.TemporaryDirectory() as temporary:
            staged = Path(temporary)
            stage_frames(frames, staged)
            if method == "detector_only":
                logits, prompt, traces = detector_only(predictor, frames, staged, detector)
            else:
                logits, prompt, traces, _ = predict_label(predictor, frames, 1, [1], recurrent, staged, False)
        if prompt is None:
            continue
        for index in range(prompt + 1, len(frames)):
            predicted = logits[index] > 0
            target = load_label_mask(frames[index].mask_path, [1]) == 1
            denominator = predicted.sum() + target.sum()
            rows.append({
                "video": video, "frame": index, "present": bool(target.any()),
                "dice": float(2 * (predicted & target).sum() / denominator) if denominator else 1.0,
                "predicted_any": bool(predicted.any()),
                "redetected": bool(traces.get(index, {}).get("redetected", False)),
            })
    return rows


def summarise(rows):
    per_video = {}
    for video in sorted({row["video"] for row in rows}):
        polyp = [row for row in rows if row["video"] == video and row["present"]]
        empty = [row for row in rows if row["video"] == video and not row["present"]]
        per_video[video] = {
            "polyp_frames": len(polyp), "empty_frames": len(empty),
            "polyp_dice": float(np.mean([row["dice"] for row in polyp])) if polyp else None,
            "lost": float(np.mean([row["dice"] < 0.1 for row in polyp])) if polyp else None,
            "empty_fp": float(np.mean([row["predicted_any"] for row in empty])) if empty else None,
        }
    mean = lambda key: float(np.mean([value[key] for value in per_video.values() if value[key] is not None]))
    polyp, empty = [row for row in rows if row["present"]], [row for row in rows if not row["present"]]
    return {
        "per_sequence": {key: mean(key) for key in ("polyp_dice", "empty_fp", "lost")},
        "pooled": {
            "polyp_dice": float(np.mean([row["dice"] for row in polyp])),
            "empty_fp": float(np.mean([row["predicted_any"] for row in empty])) if empty else None,
            "lost": float(np.mean([row["dice"] < 0.1 for row in polyp])),
        },
        "per_video": per_video,
    }


def main():
    args = parse_args()
    if args.manifest.resolve() == (ROOT / "data" / "polypgen_sequence.jsonl").resolve() and args.protocol is None:
        raise ValueError("PolypGen evaluation requires --protocol to label validation, exploratory or locked-test data.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    videos = load_manifest(args.manifest, args.split)
    if args.videos != "all":
        wanted = set(args.videos.split(","))
        videos = {video: frames for video, frames in videos.items() if video in wanted}
    if not videos:
        raise ValueError("No requested videos found in the requested split.")
    protocol = load_protocol(args.protocol) if args.protocol else None
    role = evaluation_role(protocol, videos) if protocol else None
    setup = {
        "split": args.split, "videos": sorted(videos), "methods": args.methods,
        "manifest": str(args.manifest), "sam2_cfg": args.sam2_cfg,
        "sam2_checkpoint": str(args.sam2_checkpoint), "detector": str(args.detector),
        "checkpoint_root": str(args.checkpoint_root), "full_run": args.full_run,
        "presence_only_run": args.presence_only_run, "rde_run": args.rde_run,
        "fixed_gain": args.fixed_gain, "device": args.device, "protocol_file": str(args.protocol) if args.protocol else None,
        "protocol_name": protocol.get("name") if protocol else None, "evaluation_role": role,
        "protocol": "tight first-positive-frame ground-truth box; score propagated later frames",
    }
    (args.output_dir / "run.json").write_text(json.dumps(setup, indent=2) + "\n")
    wandb_run = start_wandb(args.wandb_project, args.wandb_entity, args.wandb_name, args.wandb_mode,
                            args.wandb_base_url, setup, args.output_dir, ("evaluation", role or args.split))
    detector = PolypDetector(args.detector, args.device)
    try:
        for method in args.methods:
            rows = evaluate(method, videos, args, detector)
            summary = summarise(rows)
            (args.output_dir / f"{method}.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=2) + "\n")
            log(wandb_run, method, summary["per_sequence"], 0)
            print(method, json.dumps(summary["per_sequence"]), flush=True)
    finally:
        finish(wandb_run)


if __name__ == "__main__":
    main()
