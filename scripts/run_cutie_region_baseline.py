#!/usr/bin/env python3
"""Adapt Cutie's official scripting_demo.py to this project's frozen predictions.

No new model/attention implementation. This is an external temporal comparator,
not a SAM2 modification or evidence that boundary tokens have special utility.
"""
# ruff: noqa: E402
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "external/Cutie"
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from scripts.run_region_experiments import validate_output
from scripts.run_region_token_pilot import write_json
from training.tracking import start_wandb, log_results_artifact
sys.path.insert(0, str(AUTHOR))


def correction_mask(arm, initialized, detected, cached_mask):
    """No quality rule: initialize once, or accept every available detector mask.

    An empty initial prediction cannot initialize an object. After initialization,
    empty detector masks are valid corrections too; they are not filtered by GT.
    The two arms deliberately have different prompt budgets, recorded per frame.
    """
    if arm not in {"initial_only", "detector_refresh"}:
        raise ValueError(arm)
    if not initialized:
        return detected and bool(np.asarray(cached_mask).any())
    return arm == "detector_refresh" and detected


def build_author_model(checkpoint, device, cache):
    from cutie.model.cutie import CUTIE
    from cutie.inference.utils.args_utils import get_dataset_cfg

    # Use author defaults and the official demo's 480-pixel internal short side.
    # Avoid get_default_model(): it hard-codes CUDA and downloads GUI weights
    # into the cloned repository, rather than the requested Drive directory.
    with initialize_config_dir(version_base="1.3", config_dir=str(AUTHOR / "cutie/config")):
        cfg = compose(config_name="eval_config")
    get_dataset_cfg(cfg)
    cfg.max_internal_size = 480
    with open_dict(cfg.model):
        cfg.model.resnet_model_path = str(cache)
    cfg.weights = str(checkpoint)
    model = CUTIE(cfg).to(device).eval()
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Fail rather than silently running unmatched or random model parameters.
    model.load_state_dict(weights, strict=True)
    return model, cfg


@torch.inference_mode()
def run_sequence(model, cfg, frames, detections, cached_masks, output, arm, device, run):
    from cutie.inference.inference_core import InferenceCore

    processor = InferenceCore(model, cfg=cfg)
    initialized, prompts, reads = False, 0, [0]
    handle = model.object_transformer.register_forward_hook(
        lambda *unused: reads.__setitem__(0, reads[0] + 1))
    rows = []
    try:
        for t, frame in enumerate(frames):
            with Image.open(frame.image_path) as source:
                image = source.convert("RGB")
                width, height = image.size
                image_tensor = to_tensor(image).to(device)
            path = cached_masks / frame.video_id / f"{frame.image_path.stem}.png"
            with Image.open(path) as source:
                cached = np.asarray(source) > 0
            if cached.shape != (height, width):
                raise ValueError("Cached prediction dimensions differ from source image")
            correct = correction_mask(arm, initialized, detections[t] is not None, cached)
            propagated = initialized and not correct
            if not initialized and not correct:
                mask = np.zeros_like(cached)
            else:
                prompt = torch.from_numpy(cached.astype(np.int64)).to(device) if correct else None
                probability = processor.step(image_tensor, prompt,
                    objects=[1] if correct else None, end=t == len(frames) - 1)
                if not torch.isfinite(probability).all():
                    raise RuntimeError("Nonfinite author predictor output")
                mask = processor.output_prob_to_mask(probability).cpu().numpy() == 1
                initialized = True
                prompts += int(correct)
            # Labels enter only after inference, never in the author's processor.
            truth = load_label_mask(frame.mask_path, [1]) > 0
            row = dict(video=frame.video_id, frame=t, arm=arm, corrected=correct, propagated=propagated,
                       detected=detections[t] is not None, **calculate_scores(mask, truth),
                       framewise_dice=calculate_scores(cached, truth)["dice"],
                       cached_prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(row)
            with (output / "frames.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            destination = output / "masks" / arm / frame.video_id
            destination.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(destination / path.name)
            run.log({"frame_index": t, "video": frame.video_id, "arm": arm, "dice": row["dice"]})
            print(f"[author-cutie] {frame.video_id} {arm} {t+1}/{len(frames)}", flush=True)
    finally:
        handle.remove()
    if any(row["propagated"] for row in rows) and reads[0] == 0:
        raise RuntimeError("Object-memory readout was never exercised")
    return dict(dice=float(np.mean([r["dice"] for r in rows])), frames=len(rows),
                framewise_dice=float(np.mean([r["framewise_dice"] for r in rows])),
                correction_frames=prompts, object_memory_reads=reads[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cached-masks", type=Path, required=True)
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--max-frames", type=int, default=0, help="Nonzero: technical smoke only")
    args = parser.parse_args()
    if not set(args.sequence) <= {f"seq{i}" for i in range(1, 16)} or args.max_frames < 0:
        parser.error("Development sequences only; no held-out tuning")
    for path in [args.output_dir, args.cache_dir]:
        validate_output(path)
    expected = "a6071de6136982e396851903ab4c083a"
    if hashlib.md5(args.checkpoint.read_bytes()).hexdigest() != expected:
        raise ValueError("Checkpoint differs from author's published MD5")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(author_revision=subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=AUTHOR, text=True).strip(), adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source="https://github.com/hkchengrex/Cutie/blob/main/scripting_demo.py",
        warning="Different backbone and prompt budgets; not SAM2 region ablation or held-out confirmation",
        training=False, full_registered_study_complete=False)
    write_json(args.output_dir / "setup.json", config)
    run = start_wandb("ade-seg", "adenoid-hypertrophy", "Cutie author baseline | development",
        "online", "https://forge.coreweave.com/api/wandb", config, args.output_dir,
        ["author-baseline", "foreground-background-memory", "development"])
    try:
        model, cfg = build_author_model(args.checkpoint, args.device, args.cache_dir)
        write_json(args.output_dir / "author_config.json", OmegaConf.to_container(cfg, resolve=True))
        videos = load_manifest(args.manifest, "dev")
        summary = {}
        for video in args.sequence:
            frames = videos[video][:args.max_frames] if args.max_frames else videos[video]
            detections = json.loads((args.detections_dir / f"detections_{video}.json").read_text())
            if len(detections) != len(videos[video]):
                raise ValueError("Frozen detection count differs from sequence")
            summary[video] = {}
            for arm in ["initial_only", "detector_refresh"]:
                summary[video][arm] = run_sequence(model, cfg, frames, detections,
                    args.cached_masks, args.output_dir, arm, args.device, run)
                write_json(args.output_dir / "summary.json", summary)
                run.log({f"{video}/{arm}/dice": summary[video][arm]["dice"]})
        write_json(args.output_dir / "status.json", dict(state="complete",
            technical_smoke_only=args.max_frames > 0, full_registered_study_complete=False))
        log_results_artifact(run, args.output_dir, [str(p.relative_to(args.output_dir))
            for p in args.output_dir.rglob("*") if p.is_file() and "wandb" not in p.relative_to(args.output_dir).parts])
    except Exception as error:
        write_json(args.output_dir / "status.json", dict(state="failed", error=str(error)))
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
