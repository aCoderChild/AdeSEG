"""Fair-ish isolated Lean-SAM2 *propagation-only* CUDA benchmark.

Unlike the accuracy runner, this runner DOES NOT save masks, hash each
frame, dump TAMP selections, or synchronize per-frame telemetry. The
model/prompt/frame protocol matches the accuracy runner; a warmup pass
and synchronized repeated propagation provide separate latency evidence.

This measures end-to-end native propagation under one GPU, NOT a claim of
equal algorithmic memory complexity or portable throughput.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from types import SimpleNamespace

import torch

from adeseg.inference.lean_memory_feature_fusion import route_memory_feature_fusion
from datasets.video import load_manifest, stage_frames
from scripts.run_author_baseline import build_author_predictor, source_identity
from scripts.run_lean_memory_fusion_ablation import (
    file_sha256, get_fusion, initialize_frozen_lean_prompt,
    load_frozen_prompts, require_digest,
)
from scripts.verify_lean_prompt_provenance import verify_prompt_producer


def profile_propagation(predictor, frames, prompt, fusion, arm, device,
                        *, initialization_mode="mask_reprompt",
                        input_mode="full", warmups=1, repeats=3):
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Controlled CUDA propagation benchmark requires a real GPU")
    if type(warmups) is not int or warmups < 1 or type(repeats) is not int or repeats < 2:
        raise ValueError("Benchmark requires >=1 warmup and >=2 measured repeats")
    if getattr(predictor, "dump_retained_tokens", False):
        raise ValueError("Disable native TAMP JSONL dumps for fair timing")
    start = next(
        i for i, frame in enumerate(frames)
        if frame.frame_index == prompt["frame_index"])
    staged_frames = frames[start:]
    if len(staged_frames) < 3:
        raise ValueError("Need >=2 propagated future frames for comparable timing")
    durations, peaks, count = [], [], None
    with tempfile.TemporaryDirectory(prefix="adeseg_bench_lean_") as stage:
        # All video staging, JPEG listing, and input hashing are outside timer.
        stage_frames(staged_frames, Path(stage))
        with torch.inference_mode():
            for attempt in range(warmups + repeats):
                state = predictor.init_state(
                    video_path=stage, offload_video_to_cpu=True,
                    offload_state_to_cpu=False)
                try:
                    initialize_frozen_lean_prompt(
                        predictor, state, prompt,
                        initialization_mode=initialization_mode)
                    scope = (
                        nullcontext() if fusion is None
                        else route_memory_feature_fusion(
                            predictor, fusion, identity_control=(arm == "identity"),
                            telemetry=None, input_mode=input_mode))
                    torch.cuda.synchronize()
                    if attempt >= warmups:
                        torch.cuda.reset_peak_memory_stats()
                    began = time.perf_counter()
                    with scope:
                        seen = 0
                        for index, ids, logits in predictor.propagate_in_video(
                                state, start_frame_idx=0, reverse=False):
                            if list(ids) != [1] or logits.ndim != 4 or index != seen:
                                raise RuntimeError("Propagated frame/order contract changed")
                            seen += 1
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - began
                    if seen != len(staged_frames):
                        raise RuntimeError("Incomplete timed Lean propagation")
                    if count is not None and count != seen - 1:
                        raise RuntimeError("Benchmark frame count changed across repetitions")
                    count = seen - 1
                    if attempt >= warmups:
                        durations.append(elapsed)
                        peaks.append(int(torch.cuda.max_memory_allocated()))
                finally:
                    predictor.reset_state(state)
    if not durations or not peaks:
        raise RuntimeError("No synchronized measured repetitions")
    return {
        "future_frames_per_repeat": count,
        "warmups": warmups, "measured_repeats": repeats,
        "seconds_per_repeat": durations,
        "median_seconds": statistics.median(durations),
        "median_frames_per_second": count / statistics.median(durations),
        "gpu_peak_allocated_bytes_per_repeat": peaks,
        "median_gpu_peak_allocated_bytes": int(statistics.median(peaks)),
        "timing_scope": "native Lean forward propagation including adapter/TAMP; excludes prompt construction, staging, output PNG writes, diagnostic telemetry",
        "caveat": "One GPU and fixed one-video cohort. Profile CUDA reserved VRAM and matched interleaved order separately if making efficiency claims.",
    }


def benchmark(args):
    if args.split != "dev":
        raise ValueError("Benchmark DEV only, never tune on TEST")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Cannot overwrite benchmark receipt")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")
    manifest = args.manifest.resolve(strict=True)
    prompt_file = args.prompt_manifest.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if file_sha256(checkpoint) != require_digest(args.checkpoint_sha256):
        raise ValueError("Backbone checkpoint hash mismatch")
    repo, revision = source_identity("lean")
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain",
         "--untracked-files=no"], text=True).strip():
        raise RuntimeError("Dirty AdeSEG working tree is not benchmarkable")
    videos = load_manifest(
        manifest, "dev", require_masks=False, require_contiguous=True,
        require_jpeg=True, safe_output_ids=True)
    frozen = load_frozen_prompts(prompt_file, videos)
    producer = verify_prompt_producer(
        prompt_file, image_manifest=manifest,
        backbone_sha256=args.checkpoint_sha256,
        lean_source_revision=revision, require_verified=True,
        expected_source_videos=len(videos),
        expected_prompt_count=len(frozen), split="dev")
    eligible = []
    for name, frames in sorted(videos.items()):
        if name not in frozen:
            continue
        start = next(i for i, f in enumerate(frames)
                     if f.frame_index == frozen[name]["frame_index"])
        if len(frames) - start >= 3:
            eligible.append((name, frames))
    if not eligible:
        raise ValueError("No suitable video with >=2 propagated future frames")
    if args.video_id:
        eligible = [x for x in eligible if x[0] == args.video_id]
        if not eligible:
            raise ValueError("Requested DEV video is not eligible for timed propagation")
    name, frames = eligible[0]
    fusion, digest = get_fusion(
        args.arm, args.fusion_checkpoint, args.fusion_sha256,
        args.checkpoint_sha256, args.train_manifest_sha256,
        "cuda", input_mode=args.fusion_input_mode,
        initialization_mode=args.initialization_mode)
    predictor = build_author_predictor(
        SimpleNamespace(method="lean", checkpoint=checkpoint,
                        device="cuda"), repo)
    try:
        observed = profile_propagation(
            predictor, frames, frozen[name], fusion, args.arm, "cuda",
            initialization_mode=args.initialization_mode,
            input_mode=args.fusion_input_mode,
            warmups=args.warmups, repeats=args.repeats)
    finally:
        del predictor, fusion
    report = {
        "format": "adeseg_lean_clean_latency_v1",
        "arm": args.arm,
        "fusion_input_mode": args.fusion_input_mode,
        "initialization_mode": args.initialization_mode,
        "video_id": name, "adeseg_head": head,
        "lean_source_revision": revision,
        "backbone_sha256": args.checkpoint_sha256,
        "fusion_checkpoint_sha256": digest,
        "manifest_sha256": file_sha256(manifest),
        "prompt_manifest_sha256": file_sha256(prompt_file),
        "producer_receipt_sha256": producer["receipt_sha256"],
        "torch_version": str(torch.__version__),
        "device_name": torch.cuda.get_device_name(),
        "measurement": observed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", "utf8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--prompt-manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--arm", choices=("native", "identity", "fusion"), required=True)
    p.add_argument("--fusion-input-mode",
                   choices=("full", "no_early", "early0_only", "early1_only", "no_mask"),
                   default="full")
    p.add_argument("--initialization-mode",
                   choices=("direct_prompt", "mask_reprompt"),
                   default="mask_reprompt")
    p.add_argument("--fusion-checkpoint", type=Path)
    p.add_argument("--fusion-sha256")
    p.add_argument("--train-manifest-sha256")
    p.add_argument("--video-id")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--split", choices=("dev",), default="dev")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    print(json.dumps(benchmark(args), indent=2))


if __name__ == "__main__":
    main()
