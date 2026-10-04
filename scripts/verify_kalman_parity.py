#!/usr/bin/env python3
"""Check that training (SAM2Train + Kalman mixin) matches inference on the same clip.

Compares, frame by frame, the Kalman training path with ``KalmanMemoryVideoPredictor``
and the native-bank teacher path with MedSAM2's native predictor: mask agreement,
object scores, and (for Kalman) the mean gain. Labels are only used to place the
frame-0 box; the box is identical in both paths.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MedSAM2")]

from datasets.polypgen import get_video_frame_dir, iter_polypgen_samples
from modeling.medsam2 import build_video_predictor, load_video_frame_like_predictor
from scripts.infer import first_ground_truth_box, frame_times_from_names
from training.kalman_trainer import build_training_model, video_batch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/polypgen.yaml")
    parser.add_argument("--sequence", default="seq20")
    parser.add_argument("--clip_length", type=int, default=10)
    parser.add_argument("--kalman_checkpoint", type=Path, default=None)
    parser.add_argument("--perturb", action="store_true", help="Randomize the zero-initialized Kalman layers.")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cpu")
    parser.add_argument("--min_mask_iou", type=float, default=0.98)
    parser.add_argument("--max_score_error", type=float, default=0.05)
    parser.add_argument("--max_gain_error", type=float, default=0.01)
    return parser.parse_args()


def run_predictor(config, device, target, memory_update, clip_paths, times, box):
    predictor = build_video_predictor(
        config["sam2_cfg"], ROOT / config["sam2_checkpoint"], device=device, predictor_target=target
    )
    predictor.eval()
    if memory_update is not None:
        predictor.memory_update = memory_update
    with tempfile.TemporaryDirectory() as clip_dir:
        for path in clip_paths:
            (Path(clip_dir) / path.name).symlink_to(path.resolve())
        state = predictor.init_state(video_path=clip_dir, offload_video_to_cpu=False, offload_state_to_cpu=False)
        state.update({"kalman_enabled": True, "kalman_anchor_frame_idx": 0, "frame_times": times})
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, box=box)
        with torch.no_grad():
            masks = {idx: logits for idx, _, logits in predictor.propagate_in_video(state)}
    return masks, state["output_dict"]["non_cond_frame_outputs"]


def compare(train_frames, masks, outputs, size, with_gain):
    rows = []
    for frame_idx in range(1, len(train_frames)):
        frame, output = train_frames[frame_idx], outputs[frame_idx]
        train_mask = frame["pred_masks_high_res"][0, 0] > 0
        infer_mask = F.interpolate(masks[frame_idx].float(), size=(size, size), mode="bilinear")[0, 0] > 0
        infer_mask = infer_mask.to(train_mask.device)
        union = (train_mask | infer_mask).sum().item()
        row = {
            "frame_idx": frame_idx,
            "mask_iou": 1.0 if union == 0 else (train_mask & infer_mask).sum().item() / union,
            "score_error": abs(
                float(frame["multistep_object_score_logits"][-1].mean()) - float(output["object_score_logits"].mean())
            ),
        }
        if with_gain:
            row["gain_error"] = abs(float(frame["kalman"]["gain"].mean()) - output["kalman_trace"]["gain_mean"])
        rows.append(row)
    return rows


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = ROOT / config["data_root"]
    samples = list(iter_polypgen_samples(data_root, args.sequence))
    names = [sample.image_path.stem for sample in samples]
    prompt_idx, box = first_ground_truth_box(data_root, args.sequence, names)
    if box is None:
        raise SystemExit(f"{args.sequence} has no non-empty mask to prompt from.")
    clip = samples[prompt_idx : prompt_idx + args.clip_length]
    times = frame_times_from_names([sample.image_path.stem for sample in clip])

    model = build_training_model(config["sam2_cfg"], ROOT / config["sam2_checkpoint"], args.device)
    if args.kalman_checkpoint is not None:
        checkpoint = torch.load(args.kalman_checkpoint, map_location=args.device, weights_only=True)
        model.memory_update.load_state_dict(checkpoint["state_dict"])
    if args.perturb:
        torch.manual_seed(0)
        with torch.no_grad():
            for head in (model.memory_update.process_head[-1], model.memory_update.observation_head[-1]):
                head.weight.normal_(0, 0.05)
                head.bias.normal_(0, 0.5)
            model.memory_update.uncertainty_embedding.normal_(0, 0.05)
    model.memory_update.eval()

    size = model.image_size
    images = torch.stack([load_video_frame_like_predictor(sample.image_path, size) for sample in clip])
    masks = torch.stack([torch.from_numpy(sample.masks["polyp"]).float() for sample in clip])
    masks = F.interpolate(masks.unsqueeze(1), size=(size, size), mode="nearest").squeeze(1)
    batch = video_batch(images.to(args.device), masks.to(args.device))
    width, height = Image.open(clip[0].image_path).size
    model_box = torch.from_numpy(box) * torch.tensor([size / width, size / height] * 2)

    model.kalman_frame_times = times
    with torch.no_grad():
        backbone_out = model.prepare_prompt_inputs(model.forward_image(batch.flat_img_batch), batch)
        backbone_out["point_inputs_per_frame"][0] = {
            "point_coords": model_box.view(1, 2, 2).to(args.device),
            "point_labels": torch.tensor([[2, 3]], dtype=torch.int32, device=args.device),
        }
        student = model.forward_tracking(backbone_out, batch)
        model.kalman_enabled = False
        teacher = model.forward_tracking(backbone_out, batch)
        model.kalman_enabled = True

    clip_paths = [sample.image_path for sample in clip]
    kalman_masks, kalman_outputs = run_predictor(
        config, args.device, "modeling.kalman_memory.KalmanMemoryVideoPredictor",
        model.memory_update, clip_paths, times, box,
    )
    native_masks, native_outputs = run_predictor(config, args.device, None, None, clip_paths, times, box)
    results = {
        "kalman": compare(student, kalman_masks, kalman_outputs, size, with_gain=True),
        "native_teacher": compare(teacher, native_masks, native_outputs, size, with_gain=False),
    }
    passed = all(
        row["mask_iou"] >= args.min_mask_iou
        and row["score_error"] <= args.max_score_error
        and row.get("gain_error", 0.0) <= args.max_gain_error
        for rows in results.values() for row in rows
    )
    summary = {
        name: {
            "min_mask_iou": min(row["mask_iou"] for row in rows),
            "max_score_error": max(row["score_error"] for row in rows),
            **({"max_gain_error": max(row["gain_error"] for row in rows)} if name == "kalman" else {}),
        }
        for name, rows in results.items()
    }
    print(json.dumps({"sequence": args.sequence, "frames": len(clip), "passed": passed, "summary": summary, "rows": results}, indent=2))
    if not passed:
        raise SystemExit("Training/inference parity check failed.")


if __name__ == "__main__":
    main()
