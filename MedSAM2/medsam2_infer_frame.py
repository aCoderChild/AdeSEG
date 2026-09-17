"""Frame-independent MedSAM2 inference with YOLOv8 or dataset box prompts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_FRAME_CONFIG,
    load_json_config,
    resolve_project_path,
)
from inference.data import (
    get_data_box_dir,
    get_data_boxes,
    get_frame_names,
    get_video_frame_dir,
    get_video_name,
    get_yolo_boxes,
    load_rgb_image,
    resolve_frame_path,
    select_video_names,
)
from inference.models import build_image_predictor, load_yolo_model
from utils.mask_utils import save_masks_to_dir


@torch.inference_mode()
def frame_inference(
    predictor,
    yolo_model,
    box_source,
    base_video_dir,
    output_mask_dir,
    video_name,
    yolo_imgsz=640,
    yolo_conf=0.5,
    max_yolo_boxes_per_frame=1,
):
    """Segment every frame independently from that frame's available boxes."""
    video_dir = get_video_frame_dir(base_video_dir, video_name)
    video_output_name = get_video_name(base_video_dir, video_name)
    frame_names = get_frame_names(video_dir)
    if not frame_names:
        raise RuntimeError(f"Found no image frames in {video_dir}")

    data_box_dir = get_data_box_dir(base_video_dir, video_name)
    for frame_idx, frame_name in enumerate(frame_names):
        frame_path = resolve_frame_path(video_dir, frame_name)
        if box_source == "yolo":
            boxes = get_yolo_boxes(
                yolo_model, frame_path, yolo_imgsz, yolo_conf, max_yolo_boxes_per_frame
            )
        else:
            boxes = get_data_boxes(data_box_dir, frame_name, max_yolo_boxes_per_frame)

        image = load_rgb_image(frame_path)
        width, height = image.size
        if not boxes:
            save_masks_to_dir(
                output_mask_dir, video_output_name, frame_name, {}, height, width, False
            )
            continue

        predictor.set_image(image)
        per_obj_mask = {}
        for object_id, (box, _) in enumerate(boxes, start=1):
            masks, _, _ = predictor.predict(
                box=box,
                multimask_output=False,
                return_logits=False,
            )
            per_obj_mask[object_id] = masks[0]
        save_masks_to_dir(
            output_mask_dir,
            video_output_name,
            frame_name,
            per_obj_mask,
            height,
            width,
            False,
        )
        print(
            f"{video_output_name}: frame {frame_idx + 1}/{len(frame_names)} "
            f"({len(boxes)} box prompt(s))"
        )
    return video_output_name, frame_names


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--data_config", type=Path, default=DEFAULT_DATA_CONFIG)
    config_parser.add_argument("--model_config", type=Path, default=DEFAULT_FRAME_CONFIG)
    config_paths, _ = config_parser.parse_known_args()
    data_config = load_json_config(config_paths.data_config)
    model_config = load_json_config(config_paths.model_config)

    parser = argparse.ArgumentParser(
        description="Frame-independent MedSAM2 inference with box prompts.",
        parents=[config_parser],
    )
    parser.add_argument("--sam2_cfg", default=model_config["sam2_cfg"])
    parser.add_argument("--sam2_checkpoint", type=Path, default=resolve_project_path(model_config["sam2_checkpoint"]))
    parser.add_argument("-i", "--base_video_dir", type=Path, default=resolve_project_path(data_config["data_root"]))
    parser.add_argument("--yolo_checkpoint", type=Path, default=resolve_project_path(model_config["yolo_checkpoint"]))
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=model_config["device"])
    parser.add_argument("--box_source", choices=["yolo", "data"], default=model_config["box_source"])
    parser.add_argument("--video_list_file", type=Path, default=None)
    parser.add_argument("--seq_nums", type=int, nargs="*", default=None)
    parser.add_argument("-o", "--output_mask_dir", type=Path, required=True)
    parser.add_argument("--yolo_conf", type=float, default=model_config["yolo_conf"])
    parser.add_argument("--yolo_imgsz", type=int, default=model_config["yolo_imgsz"])
    parser.add_argument(
        "--max_yolo_boxes_per_frame",
        type=int,
        default=model_config["max_yolo_boxes_per_frame"],
    )
    parser.add_argument(
        "--apply_postprocessing",
        action=argparse.BooleanOptionalAction,
        default=model_config["apply_postprocessing"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = build_image_predictor(
        args.sam2_cfg,
        args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=args.apply_postprocessing,
    )
    yolo_model = load_yolo_model(args.yolo_checkpoint) if args.box_source == "yolo" else None
    video_names = select_video_names(args.base_video_dir, args.seq_nums, args.video_list_file)
    if not video_names:
        raise RuntimeError(f"Found no video sequences in {args.base_video_dir}")

    print(
        f"Running frame-independent MedSAM2 inference on {len(video_names)} sequence(s) "
        f"with {args.box_source} boxes."
    )
    for index, video_name in enumerate(video_names, start=1):
        print(f"{index}/{len(video_names)}: {get_video_name(args.base_video_dir, video_name)}")
        frame_inference(
            predictor=predictor,
            yolo_model=yolo_model,
            box_source=args.box_source,
            base_video_dir=args.base_video_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            yolo_imgsz=args.yolo_imgsz,
            yolo_conf=args.yolo_conf,
            max_yolo_boxes_per_frame=args.max_yolo_boxes_per_frame,
        )


if __name__ == "__main__":
    main()
