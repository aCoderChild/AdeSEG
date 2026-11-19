"""Reproducible native vs reconstructed-adapter MedSAM2 video inference.

Initialization is ORACLE FIRST-PRESENT MASK (one GT mask per video). This is
NOT automatic segmentation. All later frames are propagated without GT prompts.
Evaluate the saved predictions with scripts.evaluate_med_video; the generic\nall-frame evaluator would incorrectly count excluded oracle prompts as missing.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from adeseg.backbones.medsam2 import (
    build_verified_medsam2_latest, MEDSAM2_LATEST_SHA256
)
from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from modeling.medical_highres import (MedicalHighResolutionAdapter, attach_medical_highres_adapter)
from modeling.medical_highres.checkpoints import load_reconstructed_medical_adapter
from adeseg.research.med_video_contract import load_med_video_contract

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def oracle_first_present(frames, label_id):
    """Select first annotated present mask; never use GT to re-prompt later."""
    for index, frame in enumerate(frames):
        mask = load_label_mask(frame.mask_path, [label_id]) == label_id
        if mask.any():
            return index, mask
    return None, None

def infer_sequence(predictor, frames, output, *, label_id=1, adapter=None,
                   allow_untrained_adapter=False, record_logits=False):
    prompt, initial = oracle_first_present(frames, label_id)
    if prompt is None:
        return {"status": "unpromptable", "frames": len(frames), "predicted": 0}
    destination = output / "predictions" / frames[0].video_id
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    hook_telemetry = {}
    with tempfile.TemporaryDirectory(prefix="adeseg_medsam2_") as tmp:
        stage_frames(frames, Path(tmp))
        state = predictor.init_state(
            video_path=tmp, offload_video_to_cpu=True, offload_state_to_cpu=False
        )
        with torch.inference_mode():
            predictor.add_new_mask(
                state, frame_idx=prompt, obj_id=label_id,
                mask=torch.from_numpy(initial)
            )
        scope = (attach_medical_highres_adapter(
            predictor, adapter, allow_untrained=allow_untrained_adapter,
            telemetry=hook_telemetry)
                 if adapter is not None else nullcontext())
        with scope, torch.inference_mode():
            for index, object_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=prompt, reverse=False
            ):
                if object_ids != [label_id]:
                    raise RuntimeError("Unexpected object IDs in MedSAM2 propagation")
                if index == prompt:
                    continue  # Oracle prompt must not be counted as prediction
                if index != prompt + len(rows) + 1:
                    raise RuntimeError("Nonconsecutive or duplicate prediction index")
                if tuple(logits.shape) != (1, 1, *initial.shape):
                    raise RuntimeError("Video output shape differs from original annotation")
                mask = logits[0, 0].detach().float().cpu().numpy()
                if not np.isfinite(mask).all():
                    raise FloatingPointError("Nonfinite MedSAM2 logits")
                path = destination / (frames[index].image_path.stem + ".png")
                if path.exists():
                    raise FileExistsError(path)
                Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)
                extra = {}
                if record_logits:
                    logit_dir = output / "logits" / frames[0].video_id
                    logit_dir.mkdir(parents=True, exist_ok=True)
                    logit_path = logit_dir / (frames[index].image_path.stem + ".npy")
                    if logit_path.exists():
                        raise FileExistsError(logit_path)
                    np.save(logit_path, np.ascontiguousarray(mask, dtype=np.float32), allow_pickle=False)
                    extra = {
                        "logit_file": str(logit_path.relative_to(output)),
                        "logit_file_sha256": sha256(logit_path),
                    }
                rows.append({
                    "frame_index": frames[index].frame_index,
                    "source_image": str(frames[index].image_path),
                    "source_image_sha256": sha256(frames[index].image_path),
                    "source_gt_sha256": sha256(frames[index].mask_path),
                    "prediction_file": str(path.relative_to(output)),
                    "prediction_sha256": sha256(path),
                    "logit_sha256": hashlib.sha256(
                        np.ascontiguousarray(mask.astype(np.float32)).tobytes()
                    ).hexdigest(),
                    **extra,
                })
        if len(rows) != len(frames) - prompt - 1:
            raise RuntimeError("Prediction coverage incomplete")
        if adapter is not None and rows and hook_telemetry.get("calls", 0) < len(rows):
            raise RuntimeError("Medical adapter decoder hook did not execute on every predicted frame")
    return {
        "status": "completed", "frames": len(frames), "predicted": len(rows),
        "adapter_hook_calls": hook_telemetry.get("calls", 0),
        "adapter_nonzero_calls": hook_telemetry.get("nonzero_calls", 0),
        "prompt_source_index": frames[prompt].frame_index,
        "prompt_position": prompt, "prompt_source": "oracle_gt_mask",
        "prompt_mask_sha256": sha256(frames[prompt].mask_path),
        "prompt_image_sha256": sha256(frames[prompt].image_path),
        "predictions": rows,
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True,
                        help="Sealed MedSAM2 evaluation protocol with authenticated prompt manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu","mps","cuda"), default="cpu")
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--record-logits", action="store_true",
                        help="Store local float32 .npy tensors for independent numerical parity checks")
    parser.add_argument("--zero-initialized-adapter", action="store_true",
                        help="Verification control only: an untrained identity adapter")
    args = parser.parse_args(argv)
    if args.zero_initialized_adapter and (args.adapter_checkpoint or args.adapter_sha256):
        parser.error("Zero-adapter control must not use trained weights")
    if bool(args.adapter_checkpoint) != bool(args.adapter_sha256):
        parser.error("Adapter checkpoint and SHA256 must be supplied together")
    if args.output.exists():
        parser.error("Refusing to overwrite existing output")
    contract = load_med_video_contract(args.protocol)
    videos = contract["videos"]
    # Do not build a video predictor from an unverified or silently changed backbone.
    predictor = build_verified_medsam2_latest(
        contract["checkpoint"], model_cfg=contract["model_config"],
        device=args.device, apply_postprocessing=False
    ).eval().requires_grad_(False)
    predictor.fill_hole_area = 0
    adapter = None
    if args.adapter_checkpoint:
        adapter, _ = load_reconstructed_medical_adapter(
            args.adapter_checkpoint, expected_sha256=args.adapter_sha256,
            backbone_sha256=MEDSAM2_LATEST_SHA256,
            model_config=contract["model_config"]
        )
        adapter = adapter.to(args.device)
    elif args.zero_initialized_adapter:
        adapter = MedicalHighResolutionAdapter().to(args.device).eval()
    args.output.mkdir(parents=True)
    results = {}
    for video_id, frames in sorted(videos.items()):
        results[video_id] = infer_sequence(
            predictor, frames, args.output,
            label_id=contract["label_id"], adapter=adapter,
            allow_untrained_adapter=args.zero_initialized_adapter,
            record_logits=args.record_logits
        )
    record = {
        "protocol": "oracle_first_present_mask_post_prompt_v1",
        "split": contract["spec"]["split"],
        "manifest_sha256": contract["spec"]["dataset"]["manifest"]["sha256"],
        "research_protocol_sha256": contract["protocol_sha256"],
        "prompt_manifest_sha256": contract["prompt_manifest_sha256"],
        "evaluation_contract": contract["spec"]["evaluation"],
        "backbone_sha256": MEDSAM2_LATEST_SHA256,
        "model_config": contract["model_config"],
        "adapter_sha256": args.adapter_sha256,
        "adapter_mode": ("zero_initialized_control" if args.zero_initialized_adapter
                         else "trained" if adapter is not None else "native"),
        "label_id": contract["label_id"], "device": args.device,
        "raw_logits_recorded": bool(args.record_logits),
        "postprocessing": "no_hole_filling",
        "results": results,
    }
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "videos": len(results),
        "completed": sum(x["status"] == "completed" for x in results.values()),
        "initialization": "oracle_first_present_mask (NOT automatic)",
    }, indent=2))

if __name__ == "__main__":
    main()
