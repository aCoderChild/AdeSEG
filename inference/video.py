# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDSAM2_ROOT = PROJECT_ROOT / "MedSAM2"
for path in (PROJECT_ROOT, MEDSAM2_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adenoid.io import save_masks_to_dir, save_palette_masks_to_dir
from modeling.medsam2 import build_video_predictor, get_yolo_boxes, load_yolo_model


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "adenoid.json"


def box_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Return the tight xyxy box of a nonempty 2-D mask."""
    foreground = np.asarray(mask) > 0
    if foreground.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got {foreground.shape}.")
    if not foreground.any():
        return None
    y, x = np.where(foreground)
    return np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float32)


def _sort_key(name: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name))


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def get_video_frame_dir(base_video_dir: str | Path, video_name: str) -> Path:
    """Return an image directory for a direct video or a sequence directory."""
    base = Path(base_video_dir)
    sequence = base if video_name == "." else base / video_name
    images = sequence / "images"
    if images.is_dir():
        return images
    polypgen_images = sequence / f"images_{video_name}"
    return polypgen_images if polypgen_images.is_dir() else sequence


def get_video_name(base_video_dir: str | Path, video_name: str) -> str:
    if video_name != ".":
        return video_name
    base = Path(base_video_dir).resolve()
    return base.parent.name if base.name == "images" else base.name


def get_frame_names(video_dir: str | Path) -> list[str]:
    return sorted((path.stem for path in Path(video_dir).iterdir() if _is_image_file(path)), key=_sort_key)


def resolve_frame_path(video_dir: str | Path, frame_name: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        path = Path(video_dir) / f"{frame_name}{extension}"
        if path.is_file():
            return path
    raise FileNotFoundError(Path(video_dir) / frame_name)


def list_video_names(base_video_dir: str | Path) -> list[str]:
    base = Path(base_video_dir)
    direct_images = base / "images"
    if direct_images.is_dir() and any(_is_image_file(path) for path in direct_images.iterdir()):
        return ["."]
    sequences = [
        path.name for path in base.iterdir()
        if path.is_dir() and any(_is_image_file(frame) for frame in get_video_frame_dir(base, path.name).iterdir())
    ]
    if sequences:
        return sorted(sequences, key=_sort_key)
    return ["."] if any(_is_image_file(path) for path in base.iterdir()) else []


def get_data_boxes(base_video_dir: str | Path, video_name: str, frame_name: str, max_boxes: int):
    """Read existing ``bbox_<sequence>/<frame>.txt`` prompts when requested."""
    bbox_path = Path(base_video_dir) / video_name / f"bbox_{video_name}" / f"{frame_name}.txt"
    if not bbox_path.is_file():
        return []
    boxes = []
    for line in bbox_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected five fields in {bbox_path}: {line}")
        _, x1, y1, x2, y2 = fields
        boxes.append((np.array([x1, y1, x2, y2], dtype=np.float32), 1.0))
    return boxes[:max_boxes] if max_boxes > 0 else boxes


def select_video_names(
    base_video_dir: str | Path, seq_nums: list[int] | None = None, video_list_file: str | Path | None = None
) -> list[str]:
    if seq_nums:
        return [f"seq{number}" for number in seq_nums]
    if video_list_file:
        return [line.strip() for line in Path(video_list_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    return list_video_names(base_video_dir)


def _load_config(config_path: str | Path) -> dict[str, object]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

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
    max_yolo_boxes_per_frame=1,  # limit how many
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
                base_video_dir, video_name, frame_name, max_yolo_boxes_per_frame
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


def parse_adenoid_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_paths, _ = config_parser.parse_known_args()
    config = _load_config(config_paths.config)

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--sam2_cfg", default=config["sam2_cfg"])
    parser.add_argument("--sam2_checkpoint", type=Path, default=PROJECT_ROOT / str(config["sam2_checkpoint"]))
    parser.add_argument("-i", "--base_video_dir", type=Path, required=True)
    parser.add_argument("--yolo_checkpoint", type=Path, default=PROJECT_ROOT / str(config["yolo_checkpoint"]))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=config["device"])
    parser.add_argument("--box_source", choices=["yolo", "data"], default=config["box_source"])
    parser.add_argument("--video_list_file", type=str, default=None)
    parser.add_argument("--seq_nums", type=int, nargs="*", default=None)
    parser.add_argument("-o", "--output_mask_dir", type=Path, required=True)
    parser.add_argument("--score_thresh", type=float, default=config["score_thresh"])
    parser.add_argument("--yolo_conf", type=float, default=config["yolo_conf"])
    parser.add_argument("--yolo_imgsz", type=int, default=config["yolo_imgsz"])
    parser.add_argument("--yolo_iou_match_thresh", type=float, default=config["yolo_iou_match_thresh"])
    parser.add_argument("--max_yolo_boxes_per_frame", type=int, default=config["max_yolo_boxes_per_frame"])
    parser.add_argument("--use_all_boxes", action=argparse.BooleanOptionalAction, default=config["use_all_boxes"])
    parser.add_argument("--track_object_appearing_later_in_video", action=argparse.BooleanOptionalAction, default=config["track_object_appearing_later_in_video"])
    parser.add_argument("--select_memory_by_iou", action=argparse.BooleanOptionalAction, default=config["select_memory_by_iou"])
    parser.add_argument("--save_palette_png", action=argparse.BooleanOptionalAction, default=config["save_palette_png"])
    parser.add_argument("--apply_postprocessing", action=argparse.BooleanOptionalAction, default=config["apply_postprocessing"])
    parser.add_argument("--use_vos_optimized_video_predictor", action=argparse.BooleanOptionalAction, default=config["use_vos_optimized_video_predictor"])
    return parser.parse_args()


def run_adenoid_pipeline(args):
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
