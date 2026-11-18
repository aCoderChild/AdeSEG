"""Fail-closed preflight for a curated MedSAM2 GPU smoke sequence.

This intentionally accepts exactly ONE short DEV video. Do not run on the
clinical/test benchmark by accident or silently accept a different checkpoint.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.medsam2_latest import verify_medsam2_latest
from modeling.medical_highres.checkpoints import load_reconstructed_medical_adapter
from adeseg.backbones.medsam2 import MEDSAM2_LATEST_SHA256


def validate_curated_manifest(path, *, max_frames=24, label_id=1):
    if not 2 <= max_frames <= 128 or label_id < 1:
        raise ValueError("Invalid smoke length or label id")
    videos = load_manifest(Path(path).resolve(strict=True), "dev")
    if len(videos) != 1:
        raise ValueError("Smoke manifest must contain exactly one DEV video")
    name, frames = next(iter(videos.items()))
    if not 2 <= len(frames) <= max_frames:
        raise ValueError(f"Smoke DEV sequence must have 2..{max_frames} frames")
    present = []
    for frame in frames:
        with Image.open(frame.image_path) as image:
            image.verify()
        mask = load_label_mask(frame.mask_path, [label_id])
        present.append(bool((mask == label_id).any()))
    if not any(present[:-1]):
        raise ValueError("A first-present prompt AND a post-prompt frame are required")
    if not any(present[1:]):
        raise ValueError("At least one post-prompt GT-present frame is required for Dice")
    return {"video_id": name, "frames": len(frames),
            "prompt_position": present.index(True),
            "post_prompt_present_frames": sum(present[present.index(True)+1:])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-config", default="sam2.1_hiera_t512.yaml")
    p.add_argument("--max-frames", type=int, default=24)
    p.add_argument("--adapter-checkpoint", type=Path)
    p.add_argument("--adapter-sha256")
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        p.error("GPU smoke requires a functioning CUDA runner")
    if bool(args.adapter_checkpoint) != bool(args.adapter_sha256):
        p.error("Adapter checkpoint and expected SHA256 must be supplied together")
    config_file = Path(__file__).resolve().parents[1] / (
        "external/MedSAM2/sam2/configs") / args.model_config
    if not config_file.is_file():
        p.error(f"Unknown vendored MedSAM2 model configuration: {args.model_config}")
    info = validate_curated_manifest(args.manifest, max_frames=args.max_frames)
    verify_medsam2_latest(args.checkpoint)
    if args.adapter_checkpoint:
        load_reconstructed_medical_adapter(
            args.adapter_checkpoint, expected_sha256=args.adapter_sha256,
            backbone_sha256=MEDSAM2_LATEST_SHA256,
            model_config=args.model_config)
    print(json.dumps({
        "status": "preflight_passed", "gpu": torch.cuda.get_device_name(0),
        "checkpoint_sha256": MEDSAM2_LATEST_SHA256,
        "adapter_supplied": bool(args.adapter_checkpoint), **info,
    }, indent=2))


if __name__ == "__main__":
    main()
