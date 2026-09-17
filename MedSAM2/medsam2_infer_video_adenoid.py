# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference.config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_VIDEO_CONFIG,
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
    resolve_frame_path,
    select_video_names,
)
from inference.models import build_video_predictor, load_yolo_model
from utils.mask_utils import save_masks_to_dir, save_palette_masks_to_dir

# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"



def box_iou(box_a, box_b):
    """IoU for two xyxy boxes."""
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes_to_objects(boxes, previous_boxes, iou_thresh):
    """Greedily match current detections to existing MedSAM2 object ids."""
    candidates = []
    for box_idx, (box, confidence) in enumerate(boxes):
        for object_id, previous_box in previous_boxes.items():
            candidates.append((box_iou(box, previous_box), box_idx, object_id, confidence))
    assignments = {}
    used_objects = set()
    for iou, box_idx, object_id, confidence in sorted(candidates, reverse=True):
        if iou < iou_thresh or box_idx in assignments or object_id in used_objects:
            continue
        assignments[box_idx] = (object_id, boxes[box_idx][0], confidence)
        used_objects.add(object_id)
    return [(box_idx, *assignments[box_idx]) for box_idx in sorted(assignments)]


def save_video_segments(
    output_mask_dir,
    video_name,
    frame_names,
    video_segments,
    height,
    width,
    score_thresh,
    save_palette_png,
):
    for frame_idx, per_obj_logits in video_segments.items():
        per_obj_output_mask = {
            object_id: (logits > score_thresh).cpu().numpy()
            for object_id, logits in per_obj_logits.items()
        }
        if save_palette_png:
            save_palette_masks_to_dir(
                output_mask_dir, video_name, frame_names[frame_idx], per_obj_output_mask,
                height, width, False, DAVIS_PALETTE,
            )
        else:
            save_masks_to_dir(
                output_mask_dir, video_name, frame_names[frame_idx], per_obj_output_mask,
                height, width, False,
            )


@torch.inference_mode()
@torch.autocast(device_type="mps", dtype=torch.bfloat16)
def vos_inference(
    predictor,
    yolo_model,
    box_source,
    base_video_dir,
    output_mask_dir,
    video_name,
    output_video_name=None,
    score_thresh=0.0,
    use_all_boxes=False, # only the first available box prompt, discard all the others
    save_palette_png=False,
    yolo_imgsz=640,
    yolo_conf=0.5,
    yolo_iou_match_thresh=0.3,
    max_yolo_boxes_per_frame=1, # limit how many 
    # log_memory_selection=False,
):
    """Run MedSAM2 VOS from the first available box prompt."""
    video_dir = get_video_frame_dir(base_video_dir, video_name)
    video_output_name = get_video_name(base_video_dir, video_name)
    mask_output_name = output_video_name or video_output_name
    frame_names = get_frame_names(video_dir)
    if not frame_names:
        raise RuntimeError(f"In {video_output_name=}, found no image frames in {video_dir=}")

    inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=False)
    # inference_state["output_dict"]["log_memory_selection"] = log_memory_selection
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    data_box_dir = get_data_box_dir(base_video_dir, video_name)

    previous_boxes = {}
    first_prompt_frame_idx = None
    for input_frame_idx in range(len(frame_names)):
        frame_name = frame_names[input_frame_idx]
        if box_source == "yolo":
            boxes = get_yolo_boxes(
                yolo_model, resolve_frame_path(video_dir, frame_name), yolo_imgsz,
                yolo_conf, max_yolo_boxes_per_frame,
            )
        else:
            boxes = get_data_boxes(
                data_box_dir, frame_name, max_yolo_boxes_per_frame
            )
        if first_prompt_frame_idx is None:
            if not boxes:
                continue
            first_prompt_frame_idx = input_frame_idx
            assignments = [
                (object_id, box, confidence)
                for object_id, (box, confidence) in enumerate(boxes, 1)
            ]
        else:
            assignments = [
                (object_id, box, confidence)
                for _, object_id, box, confidence in match_boxes_to_objects(
                    boxes, previous_boxes, yolo_iou_match_thresh
                )
            ]

        for object_id, box, confidence in assignments:
            # TODO: verify which frames feed the box prompts - only the first frame with positive box
            """
            print(
                f"{video_output_name}: adding {box_source} box prompt on "
                f"frame {input_frame_idx} ({frame_name}), object {object_id}, "
                f"confidence={confidence:.4f}, box={box.tolist()}"
            )
            """
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                box=box,
            )
            previous_boxes[object_id] = box

        if not use_all_boxes:
            break

    # first positively prompted frame
    if first_prompt_frame_idx is None:
        print(
            f"Warning: in {video_output_name}, {box_source} found no boxes; "
            "saving empty masks because MedSAM2 has no prompted object memory."
        )
        video_segments = {frame_idx: {} for frame_idx in range(len(frame_names))}
        save_video_segments(
            output_mask_dir, mask_output_name, frame_names, video_segments, height, width,
            score_thresh, save_palette_png,
        )
        return video_output_name, frame_names

    video_segments = {
        frame_idx: {} for frame_idx in range(first_prompt_frame_idx)
    }
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            object_id: out_mask_logits[i].detach().float().cpu()
            for i, object_id in enumerate(out_obj_ids)
        }
    save_video_segments(
        output_mask_dir, mask_output_name, frame_names, video_segments, height, width,
        score_thresh, save_palette_png,
    )
    return video_output_name, frame_names


@torch.inference_mode()
@torch.autocast(device_type="mps", dtype=torch.bfloat16)
def vos_separate_inference_per_object(
    predictor,
    base_video_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    # log_memory_selection=False,
):
    """Run anchor-mask VOS separately for objects appearing in later frames."""
    video_dir = get_video_frame_dir(base_video_dir, video_name)
    video_output_name = get_video_name(base_video_dir, video_name)
    frame_names = get_frame_names(video_dir)
    if not frame_names:
        raise RuntimeError(f"In {video_output_name=}, found no image frames in {video_dir=}")

    inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=False)
    # inference_state["output_dict"]["log_memory_selection"] = log_memory_selection
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    # Load the first external anchor mask for each object.
    inputs_per_object = defaultdict(dict)
    anchor_mask_dir = os.path.join(base_video_dir, "anchor_masks", video_name)
    for frame_idx, frame_name in enumerate(frame_names):
        anchor_mask_path = os.path.join(anchor_mask_dir, f"{frame_name}.png")
        if not os.path.isfile(anchor_mask_path):
            continue
        with Image.open(anchor_mask_path) as anchor_image:
            anchor_mask = np.asarray(anchor_image.convert("L"), dtype=np.uint8)
        for object_id in np.unique(anchor_mask):
            if object_id == 0:
                continue
            if len(inputs_per_object[object_id]) > 0:
                continue
            object_mask = anchor_mask == object_id
            if not np.any(object_mask):
                continue
            # TODO: print the anchor mask
            print(
                f"{video_output_name}: loading anchor mask on frame {frame_idx} "
                f"({frame_name}), object {object_id}"
            )
            inputs_per_object[object_id][frame_idx] = object_mask

    if not inputs_per_object:
        print(
            f"Warning: in {video_output_name}, found no anchor masks; "
            "saving empty masks because MedSAM2 has no prompted object memory."
        )
        for frame_name in frame_names:
            save_palette_masks_to_dir(
                output_mask_dir=output_mask_dir,
                video_name=video_output_name,
                frame_name=frame_name,
                per_obj_output_mask={},
                height=height,
                width=width,
                per_obj_png_file=False,
                output_palette=DAVIS_PALETTE,
            )
        return video_output_name, frame_names

    # Run propagation separately for every automatically discovered object.
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)
    confidence_rows = []
    for object_id in object_ids:
        input_frame_inds = sorted(inputs_per_object[object_id])
        predictor.reset_state(inference_state)
        # add the mask from anchor_masks folder (1 mask per seq)
        for input_frame_idx in input_frame_inds:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=inputs_per_object[object_id][input_frame_idx],
            )

        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min(input_frame_inds),
            reverse=False,
        ):
            output_scores_per_object[object_id][out_frame_idx] = (
                out_mask_logits.cpu().numpy()
            )

        # TODO: select the frames for memory bank by IoU
        for output_type in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for frame_idx, output in inference_state["output_dict"][output_type].items():
                iou_prediction = output.get("iou_predictions")
                if iou_prediction is None:
                    continue
                confidence_rows.append({
                    "sequence": video_output_name,
                    "object_id": object_id,
                    "frame_idx": frame_idx,
                    "frame": frame_names[frame_idx],
                    "output_type": output_type,
                    "iou_prediction": float(iou_prediction.detach().float().mean().cpu()),
                })

    if confidence_rows:
        os.makedirs(output_mask_dir, exist_ok=True)
        confidence_path = os.path.join(output_mask_dir, "memory_confidence_per_frame.csv")
        write_header = not os.path.exists(confidence_path)
        with open(confidence_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(confidence_rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(confidence_rows)

    # Consolidate the independently propagated object logits into one mask.
    video_segments = {}
    for frame_idx in range(len(frame_names)):
        scores = torch.full(
            size=(len(object_ids), 1, height, width),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
        for i, object_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[object_id]:
                scores[i] = torch.from_numpy(
                    output_scores_per_object[object_id][frame_idx]
                )

        scores = predictor._apply_non_overlapping_constraints(scores)
        video_segments[frame_idx] = {
            object_id: (scores[i] > score_thresh).cpu().numpy()
            for i, object_id in enumerate(object_ids)
        }

    for frame_idx, per_obj_output_mask in video_segments.items():
        save_palette_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_output_name,
            frame_name=frame_names[frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=False,
            output_palette=DAVIS_PALETTE,
        )


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--data_config", type=str, default=DEFAULT_DATA_CONFIG)
    config_parser.add_argument("--model_config", type=str, default=DEFAULT_VIDEO_CONFIG)
    config_paths, _ = config_parser.parse_known_args()
    data_config = load_json_config(config_paths.data_config)
    model_config = load_json_config(config_paths.model_config)

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--sam2_cfg", default=model_config["sam2_cfg"])
    parser.add_argument("--sam2_checkpoint", type=str, default=resolve_project_path(model_config["sam2_checkpoint"]))
    parser.add_argument("-i", "--base_video_dir", type=str, default=resolve_project_path(data_config["data_root"]))
    parser.add_argument("--yolo_checkpoint", type=str, default=resolve_project_path(model_config["yolo_checkpoint"]))
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=model_config["device"])
    parser.add_argument("--box_source", choices=["yolo", "data"], default=model_config["box_source"])
    parser.add_argument("--video_list_file", type=str, default=None)
    parser.add_argument("--seq_nums", type=int, nargs="*", default=None)
    parser.add_argument("-o", "--output_mask_dir", type=str, required=True)
    parser.add_argument("--score_thresh", type=float, default=model_config["score_thresh"])
    parser.add_argument("--yolo_conf", type=float, default=model_config["yolo_conf"])
    parser.add_argument("--yolo_imgsz", type=int, default=model_config["yolo_imgsz"])
    parser.add_argument("--yolo_iou_match_thresh", type=float, default=model_config["yolo_iou_match_thresh"])
    parser.add_argument("--max_yolo_boxes_per_frame", type=int, default=model_config["max_yolo_boxes_per_frame"])
    parser.add_argument("--use_all_boxes", action=argparse.BooleanOptionalAction, default=model_config["use_all_boxes"])
    parser.add_argument("--track_object_appearing_later_in_video", action=argparse.BooleanOptionalAction, default=model_config["track_object_appearing_later_in_video"])
    parser.add_argument("--select_memory_by_iou", action=argparse.BooleanOptionalAction, default=model_config["select_memory_by_iou"])
    parser.add_argument("--save_palette_png", action=argparse.BooleanOptionalAction, default=model_config["save_palette_png"])
    parser.add_argument("--apply_postprocessing", action=argparse.BooleanOptionalAction, default=model_config["apply_postprocessing"])
    parser.add_argument("--use_vos_optimized_video_predictor", action=argparse.BooleanOptionalAction, default=model_config["use_vos_optimized_video_predictor"])
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = ["++model.non_overlap_masks=true"]
    if args.select_memory_by_iou:
        overrides.append("++model.select_memory_by_iou=true")
    predictor = build_video_predictor(
        args.sam2_cfg,
        args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=overrides,
        vos_optimized=args.use_vos_optimized_video_predictor,
    )
    yolo_model = load_yolo_model(args.yolo_checkpoint) if args.box_source == "yolo" else None
    video_names = select_video_names(args.base_video_dir, args.seq_nums, args.video_list_file)
    if not video_names:
        raise RuntimeError(f"Found no video sequences in {args.base_video_dir}")

    print(f"using {args.box_source} boxes as MedSAM2 video prompts")
    for n_video, video_name in enumerate(video_names, start=1):
        print(f"\n{n_video}/{len(video_names)} - running on {get_video_name(args.base_video_dir, video_name)}")
        if args.track_object_appearing_later_in_video:
            vos_separate_inference_per_object(
                predictor, args.base_video_dir, args.output_mask_dir, video_name, args.score_thresh
            )
        else:
            vos_inference(
                predictor=predictor,
                yolo_model=yolo_model,
                box_source=args.box_source,
                base_video_dir=args.base_video_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                use_all_boxes=args.use_all_boxes,
                save_palette_png=args.save_palette_png,
                yolo_imgsz=args.yolo_imgsz,
                yolo_conf=args.yolo_conf,
                yolo_iou_match_thresh=args.yolo_iou_match_thresh,
                max_yolo_boxes_per_frame=args.max_yolo_boxes_per_frame,
            )
    print(f"completed inference on {len(video_names)} videos -- output masks saved to {args.output_mask_dir}")

if __name__ == "__main__":
    main()
