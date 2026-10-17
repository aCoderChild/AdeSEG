#!/usr/bin/env python3
"""E4 image measurements using existing illumination code and torchvision RAFT.

Motion is the median flow in original-image pixels per sampled pair. PolypGen
pairs have unknown temporal gaps; they are never called velocities.
"""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import foreground_labels, load_manifest
from modeling.region_tokens import token_categories
from scripts.analyze_illumination import image_stats
from scripts.run_region_experiments import validate_output
from scripts.run_region_token_pilot import write_json
from training.tracking import start_wandb, log_results_artifact


def contrast_features(image, mask):
    labels = token_categories(mask)
    grid = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(grid.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    interior, near = labels["categories"] == 1, labels["categories"] == 2
    contrast = (
        float(np.linalg.norm(lab[interior].mean(0) - lab[near].mean(0)))
        if interior.any() and near.any()
        else None
    )
    size = float(labels["coverage"].sum())
    luminance, specular = image_stats(image)
    return {
        "contrast": contrast,
        "coverage_tokens": size,
        "size_stratum": "small" if size < 64 else "medium" if size < 512 else "large",
        "luminance": luminance,
        "specular_fraction": specular,
    }


@torch.inference_mode()
def median_motion(model, transforms, previous, current, device):
    if previous.shape != current.shape:
        raise ValueError("RAFT pair dimensions differ")
    height, width = current.shape[:2]
    # Resize to multiples of eight; preserve aspect ratio and rescale vectors
    # before computing the median norm in original-pixel coordinates.
    factor = min(1.0, 512 / max(height, width))
    size = (
        max(128, round(height * factor / 8) * 8),
        max(128, round(width * factor / 8) * 8),
    )
    images = [
        F.interpolate(
            torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float() / 255,
            size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).to(device)
        for x in (previous, current)
    ]
    first, second = transforms(*images)
    flow = model(first, second)[-1][0]
    flow[0] *= width / size[1]
    flow[1] *= height / size[0]
    return float(torch.linalg.vector_norm(flow, dim=0).median())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--dataset", required=True, choices=["polypgen", "cholecseg8k", "davis"]
    )
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights).to(args.device).eval()
    setup = {
        "stage": "E4_conditions",
        "dataset": args.dataset,
        "raft_weights": str(weights),
        "contrast": "Euclidean Lab mean foreground-interior vs near-background at 64x64",
        "motion": "median magnitude in original pixels per sampled pair",
        "full_study_complete": False,
    }
    write_json(output / "setup.json", setup)
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        "region-tokens-E4-conditions",
        "online",
        "https://forge.coreweave.com/api/wandb",
        setup,
        output,
        ["region-tokens-v1", "E4"],
    )
    try:
        videos = load_manifest(args.manifest, args.split)
        for video in args.sequence:
            previous = None
            objects = [
                n
                for n in foreground_labels(videos[video])
                if not (args.dataset == "cholecseg8k" and n == 13)
            ]
            for t, frame in enumerate(videos[video]):
                image = cv2.cvtColor(
                    cv2.imread(str(frame.image_path)), cv2.COLOR_BGR2RGB
                )
                motion = (
                    median_motion(
                        model, weights.transforms(), previous, image, args.device
                    )
                    if previous is not None
                    else None
                )
                labels = load_label_mask(frame.mask_path, list(range(1, 256)))
                for label in objects:
                    row = {
                        "video": video,
                        "frame": t,
                        "label_id": int(label),
                        "dataset": args.dataset,
                        "motion": motion,
                        "sampled_pair": args.dataset == "polypgen",
                        "class_group": "instruments"
                        if args.dataset == "cholecseg8k" and label in (5, 9)
                        else "anatomy"
                        if args.dataset == "cholecseg8k"
                        else args.dataset,
                        **contrast_features(image, labels == label),
                    }
                    with (output / "conditions.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                previous = image
                run.log({"progress/frame": t, "progress/video": video})
            log_results_artifact(
                run, output, ["conditions.jsonl", "setup.json", "tracking.json"]
            )
        write_json(
            output / "status.json",
            {
                "state": "complete",
                "stage": "E4_conditions",
                "full_study_complete": False,
            },
        )
        log_results_artifact(
            run,
            output,
            ["conditions.jsonl", "setup.json", "tracking.json", "status.json"],
        )
    except BaseException as exc:
        write_json(output / "status.json", {"state": "failed", "error": str(exc)})
        log_results_artifact(run, output, [p.name for p in output.glob("*.json*")])
        run.finish(exit_code=1)
        raise
    run.finish()


if __name__ == "__main__":
    main()
