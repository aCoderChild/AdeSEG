#!/usr/bin/env python3
"""Protocol adapter for the unmodified upstream SAM2Long and Lean-SAM2 TAMP.

Run each baseline in its own process: these repositories all own the `sam2`
Python package. Neither baseline is reimplemented in AdeSEG.
"""

# ruff: noqa: E402
import argparse
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
    upstream = ROOT / "external" / name
    sys.path[:0] = [str(upstream), str(upstream / "tools")]
    if "sam2" in sys.modules:
        raise RuntimeError(
            "Baseline must start in an isolated process before importing another sam2"
        )
    if name == "SAM2Long":
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
    parser.add_argument("--baseline", choices=["SAM2Long", "Lean-SAM2"], required=True)
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
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261017)
    # Delay all shared helper imports until the upstream `sam2` namespace has
    # been selected. Their model builder is never invoked in this process.
    predictor, settings, revision = build_upstream(
        args.baseline, args.checkpoint, args.device
    )
    from datasets.io import load_label_mask
    from datasets.video import foreground_labels, load_manifest
    from evaluation.segmentation import calculate_scores
    from scripts.audit_region_initialization import prompt_index
    from scripts.infer import stage_frames
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
            labels = foreground_labels(frames)
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
                    prefix="region_baseline_"
                ) as directory:
                    stage_frames(frames, Path(directory))
                    state = predictor.init_state(
                        video_path=directory,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
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
                    started = time.perf_counter()
                    try:
                        propagation = predictor.propagate_in_video(
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
                    finally:
                        predictor.reset_state(state)
                log_results_artifact(
                    run, output, ["rollout.jsonl", "setup.json", "tracking.json"]
                )
        write_json(
            output / "status.json",
            {"state": "complete", "stage": "E6_baseline", "full_study_complete": False},
        )
        log_results_artifact(run, output, [p.name for p in output.glob("*.json*")])
    except BaseException as exc:
        write_json(output / "status.json", {"state": "failed", "error": str(exc)})
        log_results_artifact(run, output, [p.name for p in output.glob("*.json*")])
        run.finish(exit_code=1)
        raise
    run.finish()


if __name__ == "__main__":
    main()
