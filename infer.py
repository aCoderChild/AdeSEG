#!/usr/bin/env python3
"""Foreground-aware dynamic-token MedSAM2 VOS with one YOLO box prompt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
MEDSAM2_ROOT = PROJECT_ROOT / "MedSAM2"
for path in (PROJECT_ROOT, MEDSAM2_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_DYNAMIC_CONFIG,
    load_json_config,
    resolve_project_path,
)
from inference.data import (
    get_frame_names,
    get_video_frame_dir,
    get_video_name,
    get_yolo_boxes,
    load_bgr_image,
    resolve_frame_path,
    select_video_names,
)
from inference.models import build_video_predictor, load_yolo_model
from utils.mask_utils import save_masks_to_dir


def save_diagnostics(output_mask_dir, video_name, rows):
    """Save one dynamic-token trace row per frame."""
    directory = Path(output_mask_dir) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / f"{video_name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_first_yolo_box(yolo_model, frame_path, yolo_imgsz, yolo_conf):
    """Return the highest-confidence YOLO box for the single VOS object."""
    boxes = get_yolo_boxes(yolo_model, frame_path, yolo_imgsz, yolo_conf, max_boxes=1)
    return boxes[0] if boxes else (None, None)


ABLATION_SETTINGS = {
    "native": ("native", "direct", "none", False),
    "current_only": ("recurrent", "direct", "none", False),
    "ema": ("recurrent", "fixed", "none", False),
    "ema_flow": ("recurrent", "fixed", "flow", False),
    "three_timescale": ("recurrent", "three_timescale", "none", False),
}


def load_prompt_records(path):
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@torch.inference_mode()
def vos_inference(
    predictor,
    yolo_model,
    base_video_dir,
    output_mask_dir,
    video_name,
    yolo_imgsz=640,
    yolo_conf=0.5,
    video_prompt_stride=1,
    memory_update="fixed",
    motion_alignment="none",
    fixed_memory_weight=0.5,
    token_write_rate=0.02,
    fixed_reliability=None,
    max_smooth_write=0.7,
    reliability_power=2.0,
    short_read_weight=0.5,
    reliable_read_weight=0.3,
    reliable_write_rate=0.08,
    unreliable_write_rate=0.02,
    long_term_split_power=1.5,
    unconfirmed_absence_scale=0.25,
    detector_conflict_scale=0.35,
    absence_confirmation_frames=3,
    detector_stride=1,
    memory_backend="recurrent",
    detector_recovery=True,
    detector_recovery_confidence=0.8,
    detector_recovery_iou=0.1,
    prompt_records=None,
):
    """Initialize from YOLO, with detector-guided recovery during propagation."""
    video_dir = get_video_frame_dir(base_video_dir, video_name)
    video_output_name = get_video_name(base_video_dir, video_name)
    frame_names = get_frame_names(video_dir)
    if not frame_names:
        raise RuntimeError(f"Found no image frames in {video_dir}")

    inference_state = predictor.init_state(video_path=video_dir, offload_video_to_cpu=True)
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    if detector_stride < 1:
        raise ValueError("detector_stride must be at least 1")
    if absence_confirmation_frames < 3:
        raise ValueError("absence_confirmation_frames must be at least 3")
    frames_bgr, detector_present, detector_boxes, detector_confidences = [], [], [], []
    diagnostic_rows = []
    prompt_frame_idx, prompt_box, prompt_confidence = None, None, None

    for frame_idx, frame_name in enumerate(frame_names):
        frame_path = resolve_frame_path(video_dir, frame_name)
        frames_bgr.append(load_bgr_image(frame_path))
        box, confidence = None, None
        if frame_idx % detector_stride == 0:
            box, confidence = get_first_yolo_box(
                yolo_model, frame_path, yolo_imgsz, yolo_conf
            )
            detector_present.append(box is not None)
            detector_boxes.append(box)
            detector_confidences.append(confidence)
        else:
            detector_present.append(None)
            detector_boxes.append(None)
            detector_confidences.append(None)
        if prompt_box is None and frame_idx % video_prompt_stride == 0:
            if box is not None:
                prompt_frame_idx, prompt_box, prompt_confidence = frame_idx, box, confidence

    saved_prompt = (prompt_records or {}).get(video_name)
    if saved_prompt is not None:
        prompt_frame_idx = int(saved_prompt["frame_idx"])
        if not 0 <= prompt_frame_idx < len(frame_names):
            raise ValueError(f"Saved prompt frame is outside {video_output_name}.")
        prompt_box = torch.tensor(saved_prompt["box"], dtype=torch.float32).numpy()
        if prompt_box.shape != (4,):
            raise ValueError(f"Saved prompt box for {video_output_name} must have four coordinates.")
        prompt_confidence = float(saved_prompt["confidence"])
    if prompt_box is None:
        print(f"Warning: {video_output_name}: YOLO found no box; saving empty masks.")
        for frame_idx, frame_name in enumerate(frame_names):
            save_masks_to_dir(
                output_mask_dir, video_output_name, frame_name, {}, height, width, False
            )
            diagnostic_rows.append({"frame_idx": frame_idx, "frame": frame_name, "status": "no_prompt"})
        save_diagnostics(output_mask_dir, video_output_name, diagnostic_rows)
        return video_output_name, frame_names, None

    print(
        f"{video_output_name}: adding YOLO box prompt on frame {prompt_frame_idx} "
        f"({frame_names[prompt_frame_idx]}), confidence={prompt_confidence:.4f}"
    )
    if memory_backend == "recurrent":
        inference_state.update({
            "dynamic_token_enabled": True,
            "dynamic_token_anchor_frame_idx": prompt_frame_idx,
            "dynamic_token_frames_bgr": frames_bgr,
            "dynamic_token_use_flow": motion_alignment == "flow",
            "dynamic_token_update_mode": memory_update,
            "dynamic_token_write_rate": token_write_rate,
            "dynamic_token_fixed_weight": fixed_memory_weight,
            "dynamic_token_fixed_reliability": fixed_reliability,
            "dynamic_token_max_smooth_write": max_smooth_write,
            "dynamic_token_reliability_power": reliability_power,
            "dynamic_token_short_read_weight": short_read_weight,
            "dynamic_token_reliable_read_weight": reliable_read_weight,
            "dynamic_token_reliable_write_rate": reliable_write_rate,
            "dynamic_token_unreliable_write_rate": unreliable_write_rate,
            "dynamic_token_long_term_split_power": long_term_split_power,
            "dynamic_token_unconfirmed_absence_scale": unconfirmed_absence_scale,
            "dynamic_token_detector_conflict_scale": detector_conflict_scale,
            "dynamic_token_absence_confirmation_frames": absence_confirmation_frames,
            "dynamic_token_detector_present": detector_present,
            "dynamic_token_detector_boxes": detector_boxes,
            "dynamic_token_detector_confidences": detector_confidences,
            "dynamic_token_enable_detector_recovery": detector_recovery,
            "dynamic_token_detector_recovery_confidence": detector_recovery_confidence,
            "dynamic_token_detector_recovery_iou": detector_recovery_iou,
        })
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=prompt_frame_idx,
        obj_id=1,
        box=prompt_box,
    )

    for frame_idx in range(prompt_frame_idx):
        save_masks_to_dir(
            output_mask_dir, video_output_name, frame_names[frame_idx], {}, height, width, False
        )
        diagnostic_rows.append(
            {"frame_idx": frame_idx, "frame": frame_names[frame_idx], "status": "before_prompt"}
        )
    for frame_idx, _, mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=prompt_frame_idx, reverse=False
    ):
        save_masks_to_dir(
            output_mask_dir,
            video_output_name,
            frame_names[frame_idx],
            {1: (mask_logits[0, 0] > 0).cpu().numpy()},
            height,
            width,
            False,
        )
        output_type = "cond_frame_outputs" if frame_idx == prompt_frame_idx else "non_cond_frame_outputs"
        output = inference_state["output_dict"][output_type][frame_idx]
        diagnostic_rows.append(
            {
                "frame_idx": frame_idx,
                "frame": frame_names[frame_idx],
                "status": "prompt" if frame_idx == prompt_frame_idx else "propagated",
                "prompt_confidence": prompt_confidence if frame_idx == prompt_frame_idx else "",
                "prompt_box": json.dumps(prompt_box.tolist()) if frame_idx == prompt_frame_idx else "",
                **output.get("dynamic_token_trace", {}),
            }
        )
    save_diagnostics(output_mask_dir, video_output_name, diagnostic_rows)
    return video_output_name, frame_names, {
        "frame_idx": prompt_frame_idx,
        "box": prompt_box.tolist(),
        "confidence": prompt_confidence,
    }


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--data_config", type=Path, default=DEFAULT_DATA_CONFIG)
    config_parser.add_argument("--model_config", type=Path, default=DEFAULT_DYNAMIC_CONFIG)
    config_paths, _ = config_parser.parse_known_args()
    data_config = load_json_config(config_paths.data_config)
    model_config = load_json_config(config_paths.model_config)

    parser = argparse.ArgumentParser(
        description="Foreground-aware dynamic-token MedSAM2 VOS with a YOLO box prompt.",
        parents=[config_parser],
    )
    parser.add_argument("--sam2_cfg", default=model_config["sam2_cfg"])
    parser.add_argument("--sam2_checkpoint", type=Path, default=resolve_project_path(model_config["sam2_checkpoint"]))
    parser.add_argument("-i", "--base_video_dir", type=Path, default=resolve_project_path(data_config["data_root"]))
    parser.add_argument("--yolo_checkpoint", type=Path, default=resolve_project_path(model_config["yolo_checkpoint"]))
    parser.add_argument("--seq_nums", type=int, nargs="*", default=None)
    parser.add_argument("-o", "--output_mask_dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=model_config["device"])
    parser.add_argument("--yolo_conf", type=float, default=model_config["yolo_conf"])
    parser.add_argument("--yolo_imgsz", type=int, default=model_config["yolo_imgsz"])
    parser.add_argument("--video_prompt_stride", type=int, default=model_config["video_prompt_stride"])
    parser.add_argument(
        "--memory_update",
        choices=["direct", "fixed", "adaptive", "three_timescale"],
        default=model_config["memory_update"],
    )
    parser.add_argument("--motion_alignment", choices=["none", "flow"], default=model_config["motion_alignment"])
    parser.add_argument("--memory_backend", choices=["native", "recurrent"], default="recurrent")
    parser.add_argument("--ablation", choices=list(ABLATION_SETTINGS), default=None)
    parser.add_argument("--prompt_records", type=Path, default=None)
    parser.add_argument(
        "--detector_recovery", action=argparse.BooleanOptionalAction,
        default=model_config["detector_recovery"],
    )
    parser.add_argument(
        "--detector_recovery_confidence", type=float,
        default=model_config["detector_recovery_confidence"],
    )
    parser.add_argument(
        "--detector_recovery_iou", type=float,
        default=model_config["detector_recovery_iou"],
    )
    parser.add_argument("--fixed_memory_weight", type=float, default=model_config["fixed_memory_weight"])
    parser.add_argument("--token_write_rate", type=float, default=model_config["token_write_rate"])
    parser.add_argument("--fixed_reliability", type=float, default=model_config["fixed_reliability"])
    return parser.parse_args(), model_config


def main():
    args, dynamic = parse_args()
    if args.ablation is not None:
        (
            args.memory_backend,
            args.memory_update,
            args.motion_alignment,
            args.detector_recovery,
        ) = ABLATION_SETTINGS[args.ablation]
    if args.video_prompt_stride < 1:
        raise ValueError("--video_prompt_stride must be at least 1")
    if not 0.0 <= args.fixed_memory_weight <= 1.0:
        raise ValueError("--fixed_memory_weight must be in [0, 1]")
    if not 0.0 <= args.token_write_rate <= dynamic["max_smooth_write"]:
        raise ValueError(
            f"--token_write_rate must be in [0, {dynamic['max_smooth_write']}]"
        )
    if args.fixed_reliability is not None and not 0.0 <= args.fixed_reliability <= 1.0:
        raise ValueError("--fixed_reliability must be in [0, 1]")
    for key in (
        "short_read_weight",
        "reliable_read_weight",
        "reliable_write_rate",
        "unreliable_write_rate",
        "unconfirmed_absence_scale",
        "detector_conflict_scale",
    ):
        if not 0.0 <= dynamic[key] <= 1.0:
            raise ValueError(f"{key} in the model config must be in [0, 1]")
    if dynamic["absence_confirmation_frames"] < 3 or dynamic["detector_stride"] < 1:
        raise ValueError("absence_confirmation_frames must be at least 3 and detector_stride at least 1")
    if dynamic["short_read_weight"] + dynamic["reliable_read_weight"] > 1.0:
        raise ValueError("short_read_weight + reliable_read_weight must not exceed 1")
    if dynamic["long_term_split_power"] <= 0.0:
        raise ValueError("long_term_split_power must be positive")
    if not 0.0 <= args.detector_recovery_confidence <= 1.0:
        raise ValueError("--detector_recovery_confidence must be in [0, 1]")
    if not 0.0 <= args.detector_recovery_iou <= 1.0:
        raise ValueError("--detector_recovery_iou must be in [0, 1]")
    prompt_records = load_prompt_records(args.prompt_records)

    overrides = ["++model.select_memory_by_iou=false"]
    if args.memory_backend == "recurrent":
        overrides.insert(0, f"++model._target_={dynamic['predictor_target']}")
    predictor = build_video_predictor(
        args.sam2_cfg,
        args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=False,
        hydra_overrides_extra=overrides,
    )
    if args.memory_backend == "recurrent" and (
        predictor.num_maskmem < 2 or not predictor.use_obj_ptrs_in_encoder
    ):
        raise RuntimeError("Dynamic-token inference requires native memory modules and object pointers.")
    yolo_model = load_yolo_model(args.yolo_checkpoint)

    video_names = [
        name
        for name in select_video_names(args.base_video_dir, args.seq_nums)
        if get_frame_names(get_video_frame_dir(args.base_video_dir, name))
    ]
    if not video_names:
        raise RuntimeError(f"Found no video sequences in {args.base_video_dir}")

    args.output_mask_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        PROJECT_ROOT / "modeling/fusion.py",
        PROJECT_ROOT / "modeling/reliability_gate.py",
        args.data_config,
        args.model_config,
    ]
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "data_config": str(args.data_config),
        "model_config": str(args.model_config),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sequences": video_names,
        "max_smooth_write": dynamic["max_smooth_write"],
        "reliability_power": dynamic["reliability_power"],
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "torch_version": torch.__version__,
        "inference_complete": False,
    }
    manifest_path = args.output_mask_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Running {args.memory_backend}-memory MedSAM2 VOS on {len(video_names)} sequence(s) with YOLO boxes.")
    saved_prompt_records = {}
    for index, video_name in enumerate(video_names, start=1):
        print(f"{index}/{len(video_names)}: {get_video_name(args.base_video_dir, video_name)}")
        _, _, prompt_record = vos_inference(
            predictor=predictor,
            yolo_model=yolo_model,
            base_video_dir=args.base_video_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            yolo_imgsz=args.yolo_imgsz,
            yolo_conf=args.yolo_conf,
            video_prompt_stride=args.video_prompt_stride,
            memory_update=args.memory_update,
            motion_alignment=args.motion_alignment,
            fixed_memory_weight=args.fixed_memory_weight,
            token_write_rate=args.token_write_rate,
            fixed_reliability=args.fixed_reliability,
            max_smooth_write=dynamic["max_smooth_write"],
            reliability_power=dynamic["reliability_power"],
            short_read_weight=dynamic["short_read_weight"],
            reliable_read_weight=dynamic["reliable_read_weight"],
            reliable_write_rate=dynamic["reliable_write_rate"],
            unreliable_write_rate=dynamic["unreliable_write_rate"],
            long_term_split_power=dynamic["long_term_split_power"],
            unconfirmed_absence_scale=dynamic["unconfirmed_absence_scale"],
            detector_conflict_scale=dynamic["detector_conflict_scale"],
            absence_confirmation_frames=dynamic["absence_confirmation_frames"],
            detector_stride=dynamic["detector_stride"],
            memory_backend=args.memory_backend,
            detector_recovery=args.detector_recovery,
            detector_recovery_confidence=args.detector_recovery_confidence,
            detector_recovery_iou=args.detector_recovery_iou,
            prompt_records=prompt_records,
        )
        if prompt_record is not None:
            saved_prompt_records[video_name] = prompt_record
    (args.output_mask_dir / "prompt_records.json").write_text(
        json.dumps(saved_prompt_records, indent=2) + "\n"
    )
    manifest["inference_complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
