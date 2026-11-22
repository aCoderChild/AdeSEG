"""A real-video preflight for the native/identity/learned Lean-SAM2 memory paths.

The output is explicitly a *smoke-test artifact*, never held-out Dice. No
GT mask may occur in the input. Run on a short, source-pinned DEV clip
before allocating GPU time to the full video cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from datasets.video import load_manifest
from scripts.run_author_baseline import build_author_predictor, source_identity
from scripts.run_lean_memory_fusion_ablation import (
    file_sha256, get_fusion, infer_sequence, load_frozen_prompts,
    require_digest,
)


def verify_smoke_outputs(results, *, require_anchor=False):
    if "native" not in results or "identity" not in results:
        raise ValueError("Smoke test must contain native and zero-residual identity")
    baseline, identity = results["native"], results["identity"]
    base_rows = baseline.get("predictions", [])
    zero_rows = identity.get("predictions", [])
    if not base_rows or len(base_rows) != len(zero_rows):
        raise RuntimeError("Real-video smoke test requires nonempty paired propagation")
    if any(a["frame_index"] != b["frame_index"]
           or a["prediction_sha256"] != b["prediction_sha256"]
           or a["logits_sha256"] != b["logits_sha256"]
           for a, b in zip(base_rows, zero_rows)):
        raise RuntimeError("Native/identity predicted masks or logits diverged")
    if baseline["per_frame_resident_memory"] != identity["per_frame_resident_memory"]:
        raise RuntimeError("Native/identity per-frame memory bank diverged")
    result = {
        "frames_scored": len(base_rows),
        "native_identity_logits_bitwise_equal": True,
        "native_identity_bank_equal": True,
        "trained_fusion_exercised": "fusion" in results,
    }
    if "fusion" in results:
        x = results["fusion"]
        if len(x.get("predictions", [])) != len(base_rows):
            raise RuntimeError("Learned memory fusion altered propagation frame coverage")
        if not x.get("effective_memory_writes_affecting_future", 0):
            raise RuntimeError(
                "Learned model made no BF16-visible memory write early enough "
                "to influence a subsequent scored frame")
        if require_anchor and (
            x.get("anchor_similarity_audit", {}).get("eligible_writes", 0) < 1
        ):
            raise RuntimeError("No initialized TAMP anchor observed in learned memory writes")
        result["fusion_future_effective_writes"] = x[
            "effective_memory_writes_affecting_future"]
        result["fusion_anchor_observed_writes"] = x[
            "anchor_similarity_audit"]["eligible_writes"]
    return result


def smoke(args):
    if args.split != "dev":
        raise ValueError("Smoke runs are DEV-only and never prove held-out TEST performance")
    manifest = args.manifest.resolve(strict=True)
    prompt_manifest = args.prompt_manifest.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite a smoke-test artifact")
    if file_sha256(checkpoint) != require_digest(args.checkpoint_sha256):
        raise ValueError("Wrong pinned Lean checkpoint")
    videos = load_manifest(
        manifest, "dev", require_contiguous=True,
        require_jpeg=True, safe_output_ids=True, require_masks=False)
    prompts = load_frozen_prompts(prompt_manifest, videos)
    possible = [
        (video, frames) for video, frames in sorted(videos.items())
        if video in prompts and (
            len(frames) - next(
                j for j, frame in enumerate(frames)
                if frame.frame_index == prompts[video]["frame_index"]
            ) >= 3
        )
    ]
    if not possible:
        raise ValueError("Need a DEV video with a frozen prompt and >=2 later frames")
    video, frames = possible[0]
    repo, revision = source_identity("lean")
    arms = ("native", "identity", "fusion") if args.fusion_checkpoint else (
        "native", "identity")
    output.mkdir(parents=True)
    results, resource = {}, {}
    for arm in arms:
        predictor = build_author_predictor(
            SimpleNamespace(method="lean", checkpoint=checkpoint,
                            device=args.device), repo)
        adapter, _ = get_fusion(
            arm,
            args.fusion_checkpoint if arm == "fusion" else None,
            args.fusion_sha256 if arm == "fusion" else None,
            args.checkpoint_sha256,
            args.train_manifest_sha256 if arm == "fusion" else None,
            args.device)
        folder = output / arm / "predictions" / video
        folder.mkdir(parents=True, exist_ok=False)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            results[arm] = infer_sequence(
                predictor, frames, prompts[video], folder, adapter,
                arm, args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()
                resource[arm] = {"max_memory_allocated_bytes":
                                 torch.cuda.max_memory_allocated()}
        finally:
            del predictor, adapter
    checks = verify_smoke_outputs(
        results, require_anchor=args.require_target_anchors)
    receipt = {
        "format": "adeseg_lean_memory_gpu_smoke_v1",
        "status": "smoke_verified",
        "selected_video": video, "split": "dev",
        "lean_source_revision": revision,
        "backbone_sha256": args.checkpoint_sha256,
        "manifest_sha256": file_sha256(manifest),
        "prompt_sha256": file_sha256(prompt_manifest),
        "device": args.device, "results": checks, "resources": resource,
        "warning": (
            "One-video video-propagation contract test only. "
            "No GT was read, and no accuracy, clinical, or runtime "
            "speedup claims are supported."),
    }
    (output / "smoke_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", "utf8")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--prompt-manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--fusion-checkpoint", type=Path)
    p.add_argument("--fusion-sha256")
    p.add_argument("--train-manifest-sha256")
    p.add_argument("--require-target-anchors", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    p.add_argument("--split", choices=("dev",), default="dev")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    receipt = smoke(args)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
