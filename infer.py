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

import numpy as np
import torch
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parent
MEDSAM2_ROOT = PROJECT_ROOT / "MedSAM2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MEDSAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDSAM2_ROOT))

from sam2.build_sam import build_sam2_video_predictor
from modeling.fusion import MAX_SMOOTH_WRITE, SMOOTH_RELIABILITY_POWER
from utils.mask_utils import save_masks_to_dir
from MedSAM2.medsam2_infer_video_adenoid import (
    get_frame_names,
    get_video_frame_dir,
    get_video_name,
    get_yolo_boxes,
    list_video_names,
    resolve_frame_path,
)


DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "PolypGen2021_MultiCenterData_v3"
    / "sequenceData"
    / "positive"
)
DEFAULT_SAM2_CFG = "configs/sam2.1_hiera_t512.yaml"
DEFAULT_SAM2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "MedSAM2_latest.pt"
DEFAULT_YOLO_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "polypgen_yolov8n.pt"


def save_diagnostics(output_mask_dir, video_name, rows):
    """Save one trace row per frame, including frames without a prompt."""
    directory = Path(output_mask_dir) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / f"{video_name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_first_yolo_box(
    yolo_model, frame_path, yolo_imgsz, yolo_conf
) -> tuple[np.ndarray | None, float | None]:
    """Return the highest-confidence YOLO box for the single VOS object."""
    boxes = get_yolo_boxes(
        yolo_model=yolo_model,
        frame_path=str(frame_path),
        yolo_imgsz=yolo_imgsz,
        yolo_conf=yolo_conf,
        max_boxes=1,
    )
    return boxes[0] if boxes else (None, None)


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
    memory_update="adaptive",
    motion_alignment="flow",
    fixed_memory_weight=0.5,
    token_write_rate=0.02,
    fixed_reliability=None,
):
    """Prompt the first YOLO detection once, then propagate dynamic-token VOS."""
    video_dir = get_video_frame_dir(base_video_dir, video_name)
    video_output_name = get_video_name(base_video_dir, video_name)
    frame_names = get_frame_names(video_dir)
    if not frame_names:
        raise RuntimeError(f"Found no image frames in {video_dir}")

    inference_state = predictor.init_state(video_path=video_dir, offload_video_to_cpu=True)
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    frames_bgr = []
    diagnostic_rows = []
    prompt_frame_idx, prompt_box, prompt_confidence = None, None, None

    for frame_idx, frame_name in enumerate(frame_names):
        frame_path = Path(resolve_frame_path(video_dir, frame_name))
        with Image.open(frame_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        frames_bgr.append(rgb[:, :, ::-1].copy())
        if prompt_box is None and frame_idx % video_prompt_stride == 0:
            box, confidence = get_first_yolo_box(
                yolo_model, frame_path, yolo_imgsz, yolo_conf
            )
            if box is not None:
                prompt_frame_idx = frame_idx
                prompt_box = box
                prompt_confidence = confidence

    if prompt_box is None:
        print(f"Warning: {video_output_name}: YOLO found no box; saving empty masks.")
        for frame_idx, frame_name in enumerate(frame_names):
            save_masks_to_dir(
                output_mask_dir, video_output_name, frame_name, {}, height, width, False
            )
            diagnostic_rows.append({"frame_idx": frame_idx, "frame": frame_name, "status": "no_prompt"})
        save_diagnostics(output_mask_dir, video_output_name, diagnostic_rows)
        return video_output_name, frame_names

    print(
        f"{video_output_name}: adding YOLO box prompt on frame {prompt_frame_idx} "
        f"({frame_names[prompt_frame_idx]}), confidence={prompt_confidence:.4f}"
    )
    inference_state.update({
        "dynamic_token_enabled": True,
        "dynamic_token_anchor_frame_idx": prompt_frame_idx,
        "dynamic_token_frames_bgr": frames_bgr,
        "dynamic_token_use_flow": motion_alignment == "flow",
        "dynamic_token_update_mode": memory_update,
        "dynamic_token_write_rate": token_write_rate,
        "dynamic_token_fixed_weight": fixed_memory_weight,
        "dynamic_token_fixed_reliability": fixed_reliability,
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
        diagnostic_rows.append({"frame_idx": frame_idx, "frame": frame_names[frame_idx], "status": "before_prompt"})
    for frame_idx, _, mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=prompt_frame_idx,
        reverse=False,
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
        diagnostic_rows.append({
            "frame_idx": frame_idx, "frame": frame_names[frame_idx],
            "status": "prompt" if frame_idx == prompt_frame_idx else "propagated",
            "prompt_confidence": prompt_confidence if frame_idx == prompt_frame_idx else "",
            "prompt_box": json.dumps(prompt_box.tolist()) if frame_idx == prompt_frame_idx else "",
            **output.get("dynamic_token_trace", {}),
        })
    save_diagnostics(output_mask_dir, video_output_name, diagnostic_rows)
    return video_output_name, frame_names


def main():
    parser = argparse.ArgumentParser(
        description="Foreground-aware dynamic-token MedSAM2 VOS with a YOLO box prompt."
    )
    parser.add_argument("--sam2_cfg", default=DEFAULT_SAM2_CFG)
    parser.add_argument("--sam2_checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument(
        "-i",
        "--base_video_dir",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="dataset root containing seq*/images_seq* folders, or an image folder",
    )
    parser.add_argument("--yolo_checkpoint", type=Path, default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--seq_nums", type=int, nargs="*", default=None)
    parser.add_argument("-o", "--output_mask_dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="mps")
    parser.add_argument("--yolo_conf", type=float, default=0.5)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--video_prompt_stride", type=int, default=1)
    parser.add_argument("--memory_update", choices=["direct", "fixed", "adaptive"], default="adaptive")
    parser.add_argument("--motion_alignment", choices=["none", "flow"], default="flow")
    parser.add_argument("--fixed_memory_weight", type=float, default=0.5)
    parser.add_argument(
        "--token_write_rate", type=float, default=0.02,
        help="minimum adaptive smoothing weight",
    )
    parser.add_argument("--fixed_reliability", type=float, default=None)
    args = parser.parse_args()

    if YOLO is None:
        raise RuntimeError("YOLO VOS prompting requires the ultralytics package.")
    if args.video_prompt_stride < 1:
        raise ValueError("--video_prompt_stride must be at least 1")
    if not 0.0 <= args.fixed_memory_weight <= 1.0:
        raise ValueError("--fixed_memory_weight must be in [0, 1]")
    if not 0.0 <= args.token_write_rate <= MAX_SMOOTH_WRITE:
        raise ValueError(f"--token_write_rate must be in [0, {MAX_SMOOTH_WRITE}]")
    if args.fixed_reliability is not None and not 0.0 <= args.fixed_reliability <= 1.0:
        raise ValueError("--fixed_reliability must be in [0, 1]")

    overrides = [
        "++model._target_=modeling.fusion.DynamicTokenVideoPredictor",
        "++model.select_memory_by_iou=false",
    ]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=str(args.sam2_checkpoint),
        device=args.device,
        apply_postprocessing=False,
        hydra_overrides_extra=overrides,
    )
    if predictor.num_maskmem < 2 or not predictor.use_obj_ptrs_in_encoder:
        raise RuntimeError("Dynamic-token inference requires native memory modules and object pointers.")
    yolo_model = YOLO(str(args.yolo_checkpoint))

    if args.seq_nums:
        video_names = [f"seq{sequence_number}" for sequence_number in args.seq_nums]
    else:
        # Dataset roots can also contain anchor_masks, boxes, or output folders.
        # Only directories resolving to actual image frames are input videos.
        video_names = [
            name for name in list_video_names(str(args.base_video_dir))
            if get_frame_names(get_video_frame_dir(str(args.base_video_dir), name))
        ]
    if not video_names:
        raise RuntimeError(f"Found no video sequences in {args.base_video_dir}")

    args.output_mask_dir.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), PROJECT_ROOT / "modeling/fusion.py", PROJECT_ROOT / "modeling/reliability_gate.py"]
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sequences": video_names, "max_smooth_write": MAX_SMOOTH_WRITE,
        "reliability_power": SMOOTH_RELIABILITY_POWER,
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "torch_version": torch.__version__, "inference_complete": False,
    }
    manifest_path = args.output_mask_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Running dynamic-token MedSAM2 VOS on {len(video_names)} sequence(s) with YOLO boxes.")
    for index, video_name in enumerate(video_names, start=1):
        print(f"{index}/{len(video_names)}: {get_video_name(str(args.base_video_dir), video_name)}")
        vos_inference(
            predictor=predictor,
            yolo_model=yolo_model,
            base_video_dir=str(args.base_video_dir),
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
        )
    manifest["inference_complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
