#!/usr/bin/env python3
"""Compact-state MedSAM2 VOS with one YOLO box prompt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external"
MEDSAM2_ROOT = EXTERNAL_ROOT / "MedSAM2"
for path in (PROJECT_ROOT, EXTERNAL_ROOT, MEDSAM2_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adenoid.io import save_masks_to_dir
from models.compact_memory import MEMORY_BACKENDS
from models.medsam2 import build_video_predictor, get_yolo_boxes, load_yolo_model
from validation.polypgen.dataset import (
    get_frame_names,
    get_video_frame_dir,
    get_video_name,
    resolve_frame_path,
    select_video_names,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "polypgen.json"


def load_config(config_path: Path) -> dict[str, object]:
    path = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_diagnostics(output_mask_dir, video_name, rows):
    """Save one state-update trace row per frame."""
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


def load_prompt_records(path):
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def stored_spatial_memory_bytes(inference_state, memory_backend: str) -> int:
    output_dict = inference_state["output_dict"]
    if memory_backend != "native":
        state = output_dict.get("compact_state")
        return 0 if state is None else tensor_bytes(state.features) + tensor_bytes(state.position)
    total = 0
    for output_type in ("cond_frame_outputs", "non_cond_frame_outputs"):
        for output in output_dict[output_type].values():
            total += tensor_bytes(output.get("maskmem_features"))
            total += tensor_bytes(output.get("maskmem_pos_enc"))
    return total


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
    memory_backend="fixed_ema",
    prompt_records=None,
):
    """Initialize from a YOLO box prompt and propagate through the video."""
    started = time.perf_counter()
    video_dir = get_video_frame_dir(base_video_dir, video_name)
    video_output_name = get_video_name(base_video_dir, video_name)
    frame_names = get_frame_names(video_dir)
    if not frame_names:
        raise RuntimeError(f"Found no image frames in {video_dir}")

    inference_state = predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    diagnostic_rows = []
    prompt_frame_idx, prompt_box, prompt_confidence = None, None, None

    saved_prompt = (prompt_records or {}).get(video_name)
    if saved_prompt is not None:
        prompt_frame_idx = int(saved_prompt["frame_idx"])
        if not 0 <= prompt_frame_idx < len(frame_names):
            raise ValueError(f"Saved prompt frame is outside {video_output_name}.")
        prompt_box = torch.tensor(saved_prompt["box"], dtype=torch.float32).numpy()
        if prompt_box.shape != (4,):
            raise ValueError(f"Saved prompt box for {video_output_name} must have four coordinates.")
        prompt_confidence = float(saved_prompt["confidence"])

    for frame_idx, frame_name in enumerate(frame_names):
        frame_path = resolve_frame_path(video_dir, frame_name)
        if prompt_box is None and frame_idx % video_prompt_stride == 0:
            box, confidence = get_first_yolo_box(
                yolo_model, frame_path, yolo_imgsz, yolo_conf
            )
            if box is not None:
                prompt_frame_idx, prompt_box, prompt_confidence = frame_idx, box, confidence
    if prompt_box is None:
        print(f"Warning: {video_output_name}: YOLO found no box; saving empty masks.")
        for frame_idx, frame_name in enumerate(frame_names):
            save_masks_to_dir(
                output_mask_dir, video_output_name, frame_name, {}, height, width, False
            )
            diagnostic_rows.append({"frame_idx": frame_idx, "frame": frame_name, "status": "no_prompt"})
        save_diagnostics(output_mask_dir, video_output_name, diagnostic_rows)
        return video_output_name, frame_names, None, {
            "sequence": video_output_name,
            "frames": len(frame_names),
            "elapsed_seconds": time.perf_counter() - started,
            "fps": 0.0,
            "stored_spatial_memory_bytes": 0,
        }

    print(
        f"{video_output_name}: adding YOLO box prompt on frame {prompt_frame_idx} "
        f"({frame_names[prompt_frame_idx]}), confidence={prompt_confidence:.4f}"
    )
    if memory_backend != "native":
        inference_state.update({
            "compact_state_enabled": True,
            "compact_state_anchor_frame_idx": prompt_frame_idx,
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
                **output.get("compact_state_trace", {}),
            }
        )
    save_diagnostics(output_mask_dir, video_output_name, diagnostic_rows)
    elapsed_seconds = time.perf_counter() - started
    efficiency = {
        "sequence": video_output_name,
        "frames": len(frame_names),
        "elapsed_seconds": elapsed_seconds,
        "fps": len(frame_names) / elapsed_seconds if elapsed_seconds else 0.0,
        "stored_spatial_memory_bytes": stored_spatial_memory_bytes(
            inference_state, memory_backend
        ),
    }
    return video_output_name, frame_names, {
        "frame_idx": prompt_frame_idx,
        "box": prompt_box.tolist(),
        "confidence": prompt_confidence,
    }, efficiency


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_paths, _ = config_parser.parse_known_args()
    config = load_config(config_paths.config)
    parser = argparse.ArgumentParser(
        description="Compact-state MedSAM2 VOS with a YOLO box prompt.",
        parents=[config_parser],
    )
    parser.add_argument("--sam2_cfg", default=config["sam2_cfg"])
    parser.add_argument("--sam2_checkpoint", type=Path, default=PROJECT_ROOT / str(config["sam2_checkpoint"]))
    parser.add_argument("-i", "--base_video_dir", type=Path, default=PROJECT_ROOT / str(config["data_root"]))
    parser.add_argument("--yolo_checkpoint", type=Path, default=PROJECT_ROOT / str(config["yolo_checkpoint"]))
    parser.add_argument("--seq_nums", type=int, nargs="*", default=None)
    parser.add_argument("-o", "--output_mask_dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=config["device"])
    parser.add_argument("--yolo_conf", type=float, default=config["yolo_conf"])
    parser.add_argument("--yolo_imgsz", type=int, default=config["yolo_imgsz"])
    parser.add_argument("--video_prompt_stride", type=int, default=config["video_prompt_stride"])
    parser.add_argument(
        "--memory_backend",
        choices=MEMORY_BACKENDS,
        default="fixed_ema",
    )
    parser.add_argument(
        "--fixed_ema_alpha",
        type=float,
        default=0.1,
        help="Fixed update weight for --memory_backend fixed_ema.",
    )
    parser.add_argument("--prompt_records", type=Path, default=None)
    return parser.parse_args(), config


def main():
    args, config = parse_args()
    if args.video_prompt_stride < 1:
        raise ValueError("--video_prompt_stride must be at least 1")
    if args.memory_backend == "fixed_ema" and args.fixed_ema_alpha is None:
        raise ValueError("--fixed_ema_alpha is required for fixed_ema memory.")
    if args.fixed_ema_alpha is not None and not 0.0 <= args.fixed_ema_alpha <= 1.0:
        raise ValueError("--fixed_ema_alpha must be in [0, 1].")
    prompt_records = load_prompt_records(args.prompt_records)

    overrides = ["++model.select_memory_by_iou=false"]
    if args.memory_backend != "native":
        overrides.insert(0, f"++model._target_={config['predictor_target']}")
        overrides.append(f"++model.state_update_mode={args.memory_backend}")
        if args.memory_backend == "fixed_ema":
            overrides.append(f"++model.fixed_ema_alpha={args.fixed_ema_alpha}")
    predictor = build_video_predictor(
        args.sam2_cfg,
        args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=False,
        hydra_overrides_extra=overrides,
    )
    if args.memory_backend != "native" and predictor.num_maskmem < 1:
        raise RuntimeError("State-update baselines require MedSAM2 memory features.")
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
        PROJECT_ROOT / "models/compact_memory.py",
        args.config,
    ]
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sequences": video_names,
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
    efficiency_records = []
    for index, video_name in enumerate(video_names, start=1):
        print(f"{index}/{len(video_names)}: {get_video_name(args.base_video_dir, video_name)}")
        _, _, prompt_record, efficiency = vos_inference(
            predictor=predictor,
            yolo_model=yolo_model,
            base_video_dir=args.base_video_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            yolo_imgsz=args.yolo_imgsz,
            yolo_conf=args.yolo_conf,
            video_prompt_stride=args.video_prompt_stride,
            memory_backend=args.memory_backend,
            prompt_records=prompt_records,
        )
        if prompt_record is not None:
            saved_prompt_records[video_name] = prompt_record
        efficiency_records.append(efficiency)
    (args.output_mask_dir / "prompt_records.json").write_text(
        json.dumps(saved_prompt_records, indent=2) + "\n"
    )
    (args.output_mask_dir / "efficiency.json").write_text(
        json.dumps(efficiency_records, indent=2) + "\n"
    )
    manifest["inference_complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
