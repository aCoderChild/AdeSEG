"""Create frozen TRAIN-only native/GT-conditioned memory capture pairs.

The Lean-SAM2 predictor is left untouched: this scoped observer calls its
native memory encoder once with its prediction, then separately with the
TRAIN ground-truth mask to generate an oracle *teacher target*. The GT memory
is never inserted into the inference memory bank. All source groups must
belong to an explicitly frozen TRAIN partition.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
import tempfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from modeling.lean_memory_feature_fusion import MemoryFeatureFusionConfig
from scripts.run_lean_memory_fusion_ablation import (
    load_frozen_prompts, file_sha256, require_digest,
    initialize_frozen_lean_prompt,
)
from training.split_guard import sha256, validate_train_sources


def downsample_capture_features(features, sizes, native_memory,
                                pred_masks_high_res, object_logits, config):
    """Mirror runtime feature alignment *without* increasing token count."""
    if len(features) != 3 or len(sizes) != 3 or len(config.early_channels) != 2:
        raise ValueError("Expected pinned 3-level Lean-SAM2 FPN")
    batch, channels, h, w = native_memory.shape
    if batch != 1 or channels != config.memory_channels:
        raise ValueError("Unexpected native memory shape")
    early = []
    for i, channels_expected in enumerate(config.early_channels):
        tokens, hw = features[i], sizes[i]
        if (tokens.ndim != 3 or tokens.shape[1] != 1
                or tokens.shape[2] != channels_expected
                or tokens.shape[0] != hw[0]*hw[1]):
            raise ValueError(f"Incorrect pinned FPN level {i} channels/tokens")
        feature_map = tokens.permute(1, 2, 0).reshape(1, channels_expected, *hw)
        early.append(F.interpolate(feature_map.float(), size=(h, w), mode="area")
                     .detach().cpu().clone())
    prob = F.interpolate(torch.sigmoid(pred_masks_high_res.float()),
                         size=(h, w), mode="area").clamp(1e-6, 1-1e-6)
    return {
        "early": early,
        # logit(sigmoid pooled) reconstructs online pooling at the memory grid
        "predicted_mask_logits": torch.logit(prob).detach().cpu().clone(),
        "object_score_logits": object_logits.detach().float().cpu().clone(),
    }


@contextmanager
def collect_teacher_memory(predictor, frames, output, group, *,
                           max_captures=None, teacher_presence_mode="gt_mask"):
    """Capture on non-conditioning track_step writes only, without GT feedback."""
    if teacher_presence_mode not in ("gt_mask", "predicted"):
        raise ValueError("Unknown oracle teacher object-presence policy")
    if "track_step" in vars(predictor) or "_encode_new_memory" in vars(predictor):
        raise RuntimeError("Refusing to compose with existing memory hooks")
    original_track = predictor.track_step
    original_encode = predictor._encode_new_memory
    config = MemoryFeatureFusionConfig()
    ctx = {"active": False, "frame": None, "conditioning": False}
    saved = []
    def track(self, *args, **kwargs):
        # Native code is called with keywords by SAM2VideoPredictor.
        if ctx["active"] or "frame_idx" not in kwargs or "is_init_cond_frame" not in kwargs:
            raise RuntimeError("Unexpected Lean-SAM2 tracking call contract")
        ctx.update(active=True, frame=kwargs["frame_idx"],
                   conditioning=kwargs["is_init_cond_frame"])
        try:
            return original_track(*args, **kwargs)
        finally:
            ctx.update(active=False, frame=None, conditioning=False)

    def encode(self, *args, **kwargs):
        native, position = original_encode(*args, **kwargs)
        if (not ctx["active"] or ctx["conditioning"]
                or (max_captures is not None and len(saved) >= max_captures)):
            return native, position
        if args or not {"current_vision_feats", "feat_sizes",
                        "pred_masks_high_res", "object_score_logits",
                        "is_mask_from_pts"}.issubset(kwargs):
            raise RuntimeError("Unexpected native memory encoder calling convention")
        local_idx = ctx["frame"]
        if type(local_idx) is not int or not 0 <= local_idx < len(frames):
            raise ValueError("Memory writer frame index outside staged video")
        frame = frames[local_idx]
        if native.ndim != 4 or native.shape[0] != 1:
            raise ValueError("Expected single-object native memory features")
        with torch.no_grad():
            truth = load_label_mask(frame.mask_path, [1]) == 1
            gt = torch.from_numpy(truth.astype(np.float32)).to(
                device=native.device).reshape(1, 1, *truth.shape)
            gt = F.interpolate(gt, size=kwargs["pred_masks_high_res"].shape[-2:],
                               mode="nearest")
            # GT is a teacher-generation input during TRAIN only, not an
            # inference input. Original returned memory is always native.
            gt_logit = torch.where(gt > .5, torch.full_like(gt, 16.0),
                                   torch.full_like(gt, -16.0))
            # GT-consistent oracle: a GT-present target must not silently
            # receive the model's no-object spatial embedding. Retain the
            # predicted-presence teacher only as a separately identified control.
            gt_present = bool(truth.any())
            teacher_presence_logits = (
                torch.full_like(kwargs["object_score_logits"],
                                10.0 if gt_present else -10.0)
                if teacher_presence_mode == "gt_mask"
                else kwargs["object_score_logits"]
            )
            teacher, _ = original_encode(
                current_vision_feats=kwargs["current_vision_feats"],
                feat_sizes=kwargs["feat_sizes"],
                pred_masks_high_res=gt_logit,
                object_score_logits=teacher_presence_logits,
                is_mask_from_pts=False,
            )
            extras = downsample_capture_features(
                kwargs["current_vision_feats"], kwargs["feat_sizes"],
                native, kwargs["pred_masks_high_res"],
                kwargs["object_score_logits"], config,
            )
            record = {
                "early": extras["early"],
                "native_memory": native.detach().float().cpu().clone(),
                "teacher_memory": teacher.detach().float().cpu().clone(),
                "predicted_mask_logits": extras["predicted_mask_logits"],
                "object_score_logits": extras["object_score_logits"],
                "video_id": frame.video_id,
                "frame_index": frame.frame_index,
            }
            if record["teacher_memory"].shape != record["native_memory"].shape:
                raise ValueError("Teacher memory and native memory geometry differ")
            if any(not torch.isfinite(x).all() for x in (
                    record["native_memory"], record["teacher_memory"],
                    record["predicted_mask_logits"], *record["early"])):
                raise ValueError("Non-finite memory training capture")
            path = output / "captures" / frame.video_id / f"{frame.frame_index:06d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            torch.save(record, path)
            saved.append({
                "split": "train", "source_group": group, "video_id": frame.video_id,
                "frame_index": frame.frame_index,
                "tensor_file": str(path.relative_to(output)),
                "capture_sha256": file_sha256(path),
                "capture_source_sha256": file_sha256(Path(__file__).resolve()),
                "image_sha256": file_sha256(frame.image_path),
                "annotation_sha256": file_sha256(frame.mask_path),
                "target_type": "oracle_gt_conditioned_memory_train_only",
                "teacher_presence_mode": teacher_presence_mode,
                "ground_truth_object_present": gt_present,
                "predicted_object_present": bool(
                    (kwargs["object_score_logits"].float() > 0).all().item()),

            })
        return native, position

    try:
        predictor.track_step = MethodType(track, predictor)
        predictor._encode_new_memory = MethodType(encode, predictor)
        yield saved
    finally:
        if getattr(getattr(predictor, "track_step", None), "__func__", None) is track:
            delattr(predictor, "track_step")
        if getattr(getattr(predictor, "_encode_new_memory", None), "__func__", None) is encode:
            delattr(predictor, "_encode_new_memory")


def build_captures(args):
    from scripts.run_author_baseline import build_author_predictor, source_identity

    output = args.output.resolve()
    initialization_mode = getattr(args, "initialization_mode", "mask_reprompt")
    teacher_presence_mode = getattr(args, "teacher_presence_mode", "gt_mask")
    if initialization_mode not in ("mask_reprompt", "direct_prompt"):
        raise ValueError("Unknown TRAIN prompt initialization mode")
    if teacher_presence_mode not in ("gt_mask", "predicted"):
        raise ValueError("Unknown teacher presence mode")
    if output.exists():
        raise FileExistsError("Refusing to overwrite training captures")
    if args.max_captures_per_video is not None and args.max_captures_per_video < 1:
        raise ValueError("Invalid maximum capture count")
    require_digest(args.backbone_sha256)
    if file_sha256(args.checkpoint.resolve(strict=True)) != args.backbone_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    repo, revision = source_identity("lean")
    videos = load_manifest(args.manifest.resolve(strict=True), "train",
                           require_contiguous=True, require_jpeg=True,
                           safe_output_ids=True)
    prompts = load_frozen_prompts(args.prompt_manifest, videos)
    split = json.loads(args.split_file.resolve(strict=True).read_text("utf8"))
    group_map = json.loads(args.group_map.resolve(strict=True).read_text("utf8"))
    if (not isinstance(split, dict) or set(split) != {"train", "dev", "test"}
            or any(not isinstance(split[k], list) or
                   any(not isinstance(g, str) or not g.strip() for g in split[k])
                   or len(split[k]) != len(set(split[k]))
                   for k in ("train", "dev", "test"))
            or not isinstance(group_map, dict)
            or set(group_map) != set(videos)
            or any(not isinstance(g, str) or g not in split["train"]
                   for g in group_map.values())
            or set(split["train"]) & set(split["dev"])
            or set(split["train"]) & set(split["test"])
            or set(split["dev"]) & set(split["test"])):
        raise ValueError("Unsealed/overlapping TRAIN source groups")
    output.mkdir(parents=True)
    records = []
    for video, frames in sorted(videos.items()):
        if video not in prompts:
            continue
        predictor = build_author_predictor(
            SimpleNamespace(method="lean", checkpoint=args.checkpoint, device=args.device), repo
        )
        prompt = prompts[video]
        index = next(i for i, f in enumerate(frames)
                     if f.frame_index == prompt["frame_index"])
        # Match the pinned author runner: prompt must be local frame zero.
        staged_frames = frames[index:]
        with tempfile.TemporaryDirectory(prefix="adeseg_train_memory_") as temp:
            stage_frames(staged_frames, Path(temp))
            state = predictor.init_state(video_path=temp,
                                         offload_video_to_cpu=True,
                                         offload_state_to_cpu=False)
            try:
                with torch.inference_mode():
                    initialize_frozen_lean_prompt(
                        predictor, state, prompt,
                        initialization_mode=initialization_mode)
                    with collect_teacher_memory(
                        predictor, staged_frames, output, group_map[video],
                        max_captures=args.max_captures_per_video,
                        teacher_presence_mode=teacher_presence_mode,
                    ) as captured:
                        for position, object_ids, _ in predictor.propagate_in_video(
                            state, start_frame_idx=0, reverse=False):
                            if list(object_ids) != [1] or position < 0:
                                raise RuntimeError("Invalid frozen TRAIN propagation")
                    for capture_record in captured:
                        capture_record["initialization_mode"] = initialization_mode
                    records.extend(captured)
            finally:
                predictor.reset_state(state)
        del predictor
    if not records:
        raise ValueError("No eligible memory-writing TRAIN frames captured")
    manifest = output / "train_memory_captures.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n", "utf8")
    verified = validate_train_sources(manifest, args.split_file, file_field="tensor_file")
    receipt = {
        "format": "adeseg_oracle_memory_train_capture_v1",
        "lean_source_revision": revision,
        "backbone_sha256": args.backbone_sha256,
        "source_train_manifest_sha256": sha256(args.manifest),
        "frozen_prompt_sha256": sha256(args.prompt_manifest),
        "initialization_mode": initialization_mode,
        "teacher_presence_mode": teacher_presence_mode,
        "source_group_map_sha256": sha256(args.group_map),
        "capture_manifest_sha256": sha256(manifest),
        "capture_count": len(records),
        "verified_capture_groups": verified["train_source_groups"],
        "warning": "GT masks are used ONLY for supervised memory teacher generation on TRAIN",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", "utf8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--group-map", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-captures-per-video", type=int)
    parser.add_argument("--initialization-mode",
                        choices=("mask_reprompt", "direct_prompt"),
                        default="mask_reprompt")
    parser.add_argument("--teacher-presence-mode",
                        choices=("gt_mask", "predicted"), default="gt_mask",
                        help="GT-consistent object presence in TRAIN teacher or legacy predicted presence control")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    print(json.dumps(build_captures(parser.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
