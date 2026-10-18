#!/usr/bin/env python3
"""Protocol adapters for upstream video models, without new architectures.

Run each baseline in its own process: these repositories all own the `sam2`
Python package. Neither baseline is reimplemented in AdeSEG.
"""

# ruff: noqa: E402
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_upstream(name, checkpoint, device):
    efficient = name.startswith("Efficient-SAM2")
    upstream = ROOT / "external" / ("Efficient-SAM2" if efficient else name)
    sys.path[:0] = [str(upstream), str(upstream / "tools")]
    if "sam2" in sys.modules:
        raise RuntimeError(
            "Baseline must start in an isolated process before importing another sam2"
        )
    if efficient:
        from scripts.efficient_sam2_adapter import configure, UPSTREAM_REVISION
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
        expected_patch = (ROOT / "patches/efficient-sam2-mps.patch").read_text()
        actual_patch = subprocess.check_output(["git", "diff", "--no-ext-diff", "--unified=0", "HEAD"], cwd=upstream, text=True)
        if revision != UPSTREAM_REVISION or actual_patch != expected_patch:
            raise RuntimeError("Efficient-SAM2 differs from the pinned revision plus recorded device-only port")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_b+.yaml", str(checkpoint),
            device=device, apply_postprocessing=True,
        )
        settings = configure(predictor, sparse=name.endswith("-SMR"))
    elif name == "SAM2Long":
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_b+.yaml", str(checkpoint), device=device
        )
        settings = {"num_pathway": 3, "iou_thre": 0.1, "uncertainty": 2}
    else:
        from lvos.runner import parse_args, resolve_mode_config
        from lvos_inference import build_predictor_official_aligned

        args = parse_args(
            [
                "--config",
                str(upstream / "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"),
                "--checkpoint",
                str(checkpoint),
                "--device",
                device,
                "--force_fp32",
                "--mode",
                "topk_usage_anchor",
            ]
        )
        settings = resolve_mode_config(args)
        predictor, _, _ = build_predictor_official_aligned(
            args,
            settings,
            per_obj_png_file=True,
            apply_postprocessing=True,
            require_full_lean=False,
        )
        if settings["use_swr"] or settings["enable_insurance"]:
            raise RuntimeError("TAMP-only baseline unexpectedly enabled SWR/insurance")
    predictor.fill_hole_area = 0
    return (
        predictor.eval(),
        settings,
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
        ).strip(),
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=["SAM2Long", "Lean-SAM2", "Efficient-SAM2-dense", "Efficient-SAM2-SMR"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--dataset", choices=["polypgen", "cholecseg8k", "davis"], required=True
    )
    parser.add_argument("--partition", choices=["dev", "test"], required=True)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Bounded technical smoke only; 0 evaluates all frames")
    parser.add_argument("--validation-dir", type=Path,
                        help="Completed same-device Efficient-SAM2 repeat/no-op validation")
    args = parser.parse_args()
    if args.device == "mps":
        # Fail on unsupported nondeterministic operators instead of accepting
        # a noisy dense control as evidence of an SMR effect.
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261017)
    # Delay all shared helper imports until the upstream `sam2` namespace has
    # been selected. Their model builder is never invoked in this process.
    predictor, settings, revision = build_upstream(
        args.baseline, args.checkpoint, args.device
    )
    efficient = args.baseline.startswith("Efficient-SAM2")
    if efficient and (args.dataset != "polypgen" or len(args.sequence) != 1):
        parser.error("Efficient-SAM2 adapter currently validates one binary PolypGen sequence per process")
    if efficient and not args.max_frames:
        if args.validation_dir is None:
            parser.error("Full Efficient-SAM2 runs require --validation-dir; first run bounded repeat/no-op checks")
        validation = json.loads((args.validation_dir / "validation.json").read_text())
        if not all(validation.get(key) is True for key in (
            "dense_repeat_masks_exact", "warmup_masks_exact", "native_sparse_reads_verified"
        )) or validation.get("device") != args.device or validation.get("checkpoint_sha256") != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
            parser.error("Efficient-SAM2 repeat/no-op gate failed or device/checkpoint differs")
    if args.max_frames and (args.max_frames < 10 or args.partition != "dev"):
        parser.error("Bounded smoke requires dev partition and at least 10 frames")
    from datasets.io import load_label_mask
    from datasets.video import foreground_labels, load_manifest, stage_frames
    from evaluation.segmentation import calculate_scores
    from scripts.audit_region_initialization import prompt_index
    from scripts.run_region_experiments import (
        digest,
        require_gate,
        validate_method_split,
        validate_output,
    )
    from scripts.run_region_token_pilot import metrics, write_json
    from training.tracking import start_wandb, log_results_artifact

    output = validate_output(args.output_dir)
    gate = require_gate(args.gate_dir)
    raw = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    sources = {r["video_id"]: r.get("source_video", r["video_id"]) for r in raw}
    validate_method_split(
        args.dataset, args.partition == "test", args.sequence, sources, raw
    )
    videos = load_manifest(args.manifest, args.split)
    output.mkdir(parents=True, exist_ok=False)
    setup = {
        "stage": "E6_baseline",
        "baseline": args.baseline,
        "upstream_git_revision": revision,
        "settings": settings,
        "checkpoint_sha256": digest(args.checkpoint),
        "gate": gate,
        "partition": args.partition,
        "manifest_sha256": digest(args.manifest),
        "full_study_complete": False,
        "max_frames": args.max_frames,
        "technical_smoke_only": args.max_frames > 0,
        "model_implementation": "upstream_with_device_port" if efficient else "upstream",
        "adapter_sha256": digest(ROOT / "scripts/efficient_sam2_adapter.py") if efficient else None,
        "runner_sha256": digest(Path(__file__)),
        "device": args.device,
        "torch_version": torch.__version__,
        "autocast": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "state_offload_to_cpu": not (efficient and args.device == "mps"),
        "upstream_device_patch_sha256": digest(ROOT / "patches/efficient-sam2-mps.patch") if efficient else None,
        "region_labels_affect_selection": False,
        "measurement_taxonomy": "predicted mask mixed-coverage 0.1/0.9; boundary band 1; display only" if efficient else None,
    }
    write_json(output / "setup.json", setup)
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        f"region-tokens-{args.baseline}",
        "online",
        "https://forge.coreweave.com/api/wandb",
        setup,
        output,
        ["region-tokens-v1", "E6", args.baseline],
    )
    protocol = json.loads((ROOT / "protocols/region_tokens_v1_1.json").read_text())
    try:
        for video in args.sequence:
            frames = videos[video]
            if args.max_frames:
                frames = frames[:args.max_frames]
            labels = foreground_labels(frames)
            if efficient and labels != [1]:
                raise ValueError("Efficient-SAM2 adapter requires a single foreground label 1")
            for label in (
                n for n in labels if not (args.dataset == "cholecseg8k" and n == 13)
            ):
                try:
                    prompt = prompt_index(frames, protocol, label, labels)
                except ValueError:
                    continue
                if len(frames) <= prompt + 7:
                    continue
                with tempfile.TemporaryDirectory(
                    prefix="region_baseline_", dir=output
                ) as directory:
                    stage_frames(frames, Path(directory))
                    state = predictor.init_state(
                        video_path=directory,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=not (efficient and args.device == "mps"),
                    )
                    if args.baseline == "SAM2Long":
                        state.update(settings)
                    prompt_mask = (
                        load_label_mask(frames[prompt].mask_path, labels) == label
                    )
                    predictor.add_new_mask(
                        state,
                        frame_idx=prompt,
                        obj_id=1,
                        mask=torch.from_numpy(prompt_mask),
                    )
                    previous_prediction, previous_target = None, None
                    predictions = {prompt: prompt_mask}
                    if efficient:
                        from scripts.efficient_sam2_adapter import selection_rows, save_overlay
                        from modeling.region_tokens import token_categories
                    started = time.perf_counter()
                    try:
                        propagate = predictor.propagate_in_video_wj if efficient else predictor.propagate_in_video
                        propagation = propagate(
                            state, start_frame_idx=prompt
                        )
                        if args.baseline == "SAM2Long":
                            ids, masks = propagation
                            propagation = (
                                (prompt + i, ids, value)
                                for i, value in enumerate(masks)
                            )
                        for t, _, logits in propagation:
                            target = (
                                load_label_mask(frames[t].mask_path, labels) == label
                            )
                            prediction = logits[0, 0].float().cpu().numpy() > 0
                            if efficient and args.baseline.endswith("-SMR"):
                                queue = predictor.mem_info["mem_masks_stack"].get(0, [])
                                if queue and t - 1 in predictions:
                                    categories = token_categories(predictions[t - 1])["categories"]
                                    layers = [m.detach().cpu().numpy().copy() for m in queue[-1]]
                                    measurements = selection_rows(layers, categories)
                                    with (output / "retained_tokens.jsonl").open("a") as stream:
                                        for measurement in measurements:
                                            stream.write(json.dumps(dict(video=video, label_id=label, target_frame=t,
                                                memory_frame=t-1, category_source="previous_prediction", **measurement)) + "\n")
                                    if t <= prompt + 4:
                                        for measurement in measurements:
                                            save_overlay(frames[t-1].image_path, categories, measurement["retained_indices"],
                                                output / "visuals" / f"{video}_obj{label}_memory{t-1}_layer{measurement['layer']}.png")
                            predictions[t] = prediction
                            if t > prompt:
                                void = (
                                    load_label_mask(frames[t].mask_path, labels) == 13
                                    if args.dataset == "cholecseg8k"
                                    else None
                                )
                                stability = (
                                    None
                                    if previous_prediction is None
                                    else abs(
                                        calculate_scores(
                                            prediction, previous_prediction
                                        )["iou"]
                                        - calculate_scores(target, previous_target)[
                                            "iou"
                                        ]
                                    )
                                )
                                row = {
                                    "video": video,
                                    "source_video": sources[video],
                                    "label_id": label,
                                    "frame": int(t),
                                    "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                                    "dataset": args.dataset,
                                    "split": args.split,
                                    "category": args.baseline,
                                    "bank": None,
                                    "seed": 0,
                                    "state": "skipped"
                                    if void is not None and void.all()
                                    else "complete",
                                    "mode": "persistent",
                                    "metrics": metrics(
                                        logits[:, :1],
                                        target,
                                        void,
                                        decoder_logits=False,
                                    ),
                                    "stability_error": stability,
                                }
                                with (output / "rollout.jsonl").open("a") as stream:
                                    stream.write(
                                        json.dumps(row, allow_nan=False) + "\n"
                                    )
                            previous_prediction, previous_target = prediction, target
                        run.log(
                            {
                                "cost/fps": (len(frames) - prompt - 1)
                                / (time.perf_counter() - started),
                                "progress/video": video,
                                "progress/label": label,
                            }
                        )
                        if efficient:
                            sparse_reads = [list(layer.cross_attn_image.drop_ratio_log)
                                            for layer in predictor.memory_attention.layers]
                            if args.baseline.endswith("-SMR") and not all(sparse_reads):
                                raise RuntimeError("No actual sparse memory reads observed")
                            write_json(output / "native_sparse_reads.json", {
                                "drop_ratios_by_layer": sparse_reads,
                                "selection_is_native": True,
                                "last_frame_selection": "upstream clears cache before final yield; not reconstructed",
                            })
                    finally:
                        predictor.reset_state(state)
                log_results_artifact(
                    run, output, ["rollout.jsonl", "setup.json", "tracking.json"]
                )
        write_json(
            output / "status.json",
            {"state": "complete", "stage": "E6_baseline", "full_study_complete": False},
        )
        if efficient and args.baseline.endswith("-SMR"):
            if not (output / "retained_tokens.jsonl").exists():
                raise RuntimeError("No native sparse selections observed; cannot count this as SMR")
            log_results_artifact(run, output, [str(p.relative_to(output)) for p in output.rglob("*")
                if p.is_file() and "wandb" not in p.relative_to(output).parts])
        log_results_artifact(run, output, [p.name for p in output.glob("*.json*")])
    except BaseException as exc:
        write_json(output / "status.json", {"state": "failed", "error": str(exc)})
        log_results_artifact(run, output, [p.name for p in output.glob("*.json*")])
        run.finish(exit_code=1)
        raise
    run.finish()


if __name__ == "__main__":
    main()
