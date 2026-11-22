"""Produce frozen YOLO -> FGPI/box-only -> Lean-SAM2 automatic mask prompts.

This is the *same* pinned Lean-SAM2 Hiera-B+ backbone as the downstream
multiscale memory-write experiment. Only image-only manifest rows and
authenticated YOLO detections are read; no ground-truth paths are accepted.
The resulting first-frame binary mask is frozen and independently hashed.
No later-frame propagation is run by this prompt producer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
from PIL import Image
import torch

from adeseg.modules.fgpi import FGPI, FGPIConfig, feature_guided_prompt
from datasets.video import load_manifest, stage_frames
from scripts.detection_provenance import verify_detection_export
from scripts.infer_fgpi_video import load_detector_records
from scripts.run_lean_memory_fusion_ablation import file_sha256, require_digest


PROTOCOL = "adeseg_lean_detector_fgpi_frozen_initial_mask_v1"


def produce_sequence(predictor, frames, detections, output, *, mode="fgpi",
                     confidence_gate=0.0, positives=1, negatives=1,
                     max_anchors=4):
    """Return prompt row and diagnostic; no GT, no previous/future mask is read."""
    if mode not in ("fgpi", "box_only"):
        raise ValueError("Only box_only and fgpi are legal prompt modes")
    candidates = [x for x in detections if x["confidence"] >= confidence_gate]
    if not candidates:
        return None, {"status": "no_detection", "source_frames": len(frames)}
    selected = candidates[0]  # Frozen first eligible detection; no GT selection.
    positions = {x.frame_index: idx for idx, x in enumerate(frames)}
    start = positions[selected["frame_index"]]
    if frames[start].mask_path is not None:
        raise ValueError("GT-bearing manifest cannot enter Lean prompt generation")
    state = None
    with tempfile.TemporaryDirectory(prefix="adeseg_lean_fgpi_prompt_") as stage:
        stage_frames(frames[start:], Path(stage))
        with torch.inference_mode():
            try:
                state = predictor.init_state(
                    video_path=stage, offload_video_to_cpu=True,
                    offload_state_to_cpu=False)
                height = int(state["video_height"])
                width = int(state["video_width"])
                with Image.open(frames[start].image_path) as source_image:
                    if source_image.size != (width, height):
                        raise RuntimeError("Lean frame preprocessing changed source-image geometry")
                image_hw = (height, width)
                if mode == "fgpi":
                    _, backbone, *_ = predictor._get_image_feature(
                        state, frame_idx=0, batch_size=1)
                    levels = backbone.get("backbone_fpn") if isinstance(backbone, dict) else None
                    if (not isinstance(levels, (tuple, list))
                            or len(levels) != 3
                            or [int(v.shape[1]) for v in levels] != [32, 64, 256]
                            or any(not torch.isfinite(v).all() for v in levels)):
                        raise RuntimeError("Pinned Lean early-FPN 32/64/256 contract changed")
                    # Use the same projected early image feature as the
                    # downstream memory-fusion adapter, not MedSAM2 features.
                    prompt = FGPI(FGPIConfig(
                        positive_points=positives,
                        negative_points=negatives,
                        max_pseudo_anchors=max_anchors)).construct(
                            levels[0], selected["box_xyxy"], image_hw,
                            detector_confidence=selected["confidence"])
                else:
                    prompt = feature_guided_prompt(
                        None, selected["box_xyxy"], image_hw,
                        mode="box_only",
                        detector_confidence=selected["confidence"],
                        min_detector_confidence=confidence_gate)
                local_frame, object_ids, logits = FGPI.initialize(
                    predictor, state, 0, 1, prompt)
                if (local_frame != 0 or list(object_ids) != [1]
                        or not isinstance(logits, torch.Tensor)
                        or tuple(logits.shape) != (1, 1, height, width)
                        or not torch.isfinite(logits).all()):
                    raise RuntimeError("Lean first-frame automatic prompt output contract changed")
                mask = logits[0, 0].detach().float().cpu().numpy() > 0
            finally:
                if state is not None:
                    predictor.reset_state(state)
    if not mask.any():
        return None, {
            "status": "empty_initial_mask",
            "source_frames": len(frames),
            "selected_frame": selected["frame_index"],
            "mode": mode,
        }
    name = frames[0].video_id
    relative_mask = Path("automatic_masks") / name / (
        f"{selected['frame_index']:06d}.png")
    destination = output / relative_mask
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to overwrite frozen first-frame mask")
    Image.fromarray(mask.astype(np.uint8) * 255).save(destination)
    row = {
        "video_id": name,
        "frame_index": selected["frame_index"],
        "mask_path": relative_mask.as_posix(),
        "image_sha256": selected["image_sha256"],
        "mask_sha256": file_sha256(destination),
        "prompt_origin": "automatic_mask",
    }
    return row, {
        "status": "completed",
        "source_frames": len(frames),
        "selected_frame": selected["frame_index"],
        "selected_box": selected["box_xyxy"],
        "detector_confidence": selected["confidence"],
        "selected_image_sha256": selected["image_sha256"],
        "prompt_mode": mode,
        "prompt_details": prompt,
        "initial_mask_sha256": row["mask_sha256"],
        "initial_foreground_pixels": int(mask.sum()),
        "post_prompt_frames": len(frames) - start - 1,
    }


def export_lean_prompts(manifest, detections, detector_checkpoint,
                        detector_sha256, backbone, backbone_sha256,
                        output, *, split="dev", mode="fgpi",
                        confidence_gate=0.0, positives=1, negatives=1,
                        max_anchors=4, device="cuda"):
    from scripts.run_author_baseline import build_author_predictor, source_identity
    manifest = Path(manifest).resolve(strict=True)
    detections = Path(detections).resolve(strict=True)
    detector_checkpoint = Path(detector_checkpoint).resolve(strict=True)
    backbone = Path(backbone).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite Lean automatic prompt export")
    if (not 0 <= confidence_gate <= 1
            or type(positives) is not int or not 1 <= positives <= 8
            or type(negatives) is not int or not 1 <= negatives <= 8
            or type(max_anchors) is not int or not 1 <= max_anchors <= 16):
        raise ValueError("Invalid detector gate or FGPI point budgets")
    if file_sha256(detector_checkpoint) != require_digest(detector_sha256):
        raise ValueError("YOLO detector checkpoint SHA256 mismatch")
    if file_sha256(backbone) != require_digest(backbone_sha256):
        raise ValueError("Lean-SAM2 checkpoint SHA256 mismatch")
    videos = load_manifest(
        manifest, split, require_contiguous=True, require_jpeg=True,
        safe_output_ids=True, require_masks=False)
    detection_receipt = verify_detection_export(
        manifest, detections, detector_checkpoint, split=split,
        image_only_manifest=True)
    frozen_detections = load_detector_records(detections, videos)
    repo, revision = source_identity("lean")
    output.mkdir(parents=True)
    rows, diagnostics = [], {}
    for name, frames in sorted(videos.items()):
        predictor = build_author_predictor(
            SimpleNamespace(method="lean", checkpoint=backbone, device=device),
            repo)
        try:
            row, diagnostic = produce_sequence(
                predictor, frames, frozen_detections.get(name, []), output,
                mode=mode, confidence_gate=confidence_gate,
                positives=positives, negatives=negatives,
                max_anchors=max_anchors)
        finally:
            del predictor
        diagnostics[name] = diagnostic
        if row is not None:
            rows.append(row)
    # A cohort with zero promptable videos is a real detector/initialization
    # failure. Persist its zero-coverage receipt for independent assessment;
    # downstream memory inference still refuses an empty prompt manifest.
    prompts_file = output / "frozen_prompts.jsonl"
    with prompts_file.open("x", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    completed = sum(x["status"] == "completed" for x in diagnostics.values())
    receipt = {
        "protocol": PROTOCOL,
        "producer_source_sha256": file_sha256(Path(__file__).resolve()),
        "mode": mode, "split": split,
        "image_only_manifest_sha256": file_sha256(manifest),
        "detections_sha256": file_sha256(detections),
        "detection_metadata_sha256": detection_receipt["metadata_sha256"],
        "detector_checkpoint_sha256": detector_sha256,
        "lean_checkpoint_sha256": backbone_sha256,
        "lean_source_revision": revision,
        "confidence_gate": confidence_gate,
        "positive_points": positives, "negative_points": negatives,
        "max_pseudo_anchors": max_anchors,
        "source_video_count": len(videos),
        "promptable_video_count": completed,
        "no_detection_count": sum(
            x["status"] == "no_detection" for x in diagnostics.values()),
        "empty_initial_mask_count": sum(
            x["status"] == "empty_initial_mask" for x in diagnostics.values()),
        "promptable_video_fraction": completed / len(videos),
        "usable_for_memory_propagation": completed > 0,
        "frozen_prompt_manifest_sha256": file_sha256(prompts_file),
        "per_video": diagnostics,
        "warning": "No inference GT consumed. Frozen masks are automatic, not oracle. "
                   "Fused-video propagation is a separate authenticated run.",
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", "utf8")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True,
                   help="STRICT image-only JSONL with no mask field")
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--detector-checkpoint", type=Path, required=True)
    p.add_argument("--detector-sha256", required=True)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Same pinned Lean-SAM2 Hiera-B+ weights as memory runner")
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    p.add_argument("--mode", choices=("box_only", "fgpi"), required=True)
    p.add_argument("--confidence-gate", type=float, default=0.0)
    p.add_argument("--positives", type=int, default=1)
    p.add_argument("--negatives", type=int, default=1)
    p.add_argument("--max-anchors", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    a = p.parse_args(argv)
    receipt = export_lean_prompts(
        a.manifest, a.detections, a.detector_checkpoint,
        a.detector_sha256, a.checkpoint, a.checkpoint_sha256,
        a.output, split=a.split, mode=a.mode,
        confidence_gate=a.confidence_gate,
        positives=a.positives, negatives=a.negatives,
        max_anchors=a.max_anchors, device=a.device)
    print(json.dumps({
        "protocol": receipt["protocol"],
        "mode": receipt["mode"],
        "promptable": receipt["promptable_video_count"],
        "total": receipt["source_video_count"],
        "coverage": receipt["promptable_video_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
