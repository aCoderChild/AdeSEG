"""Run one pinned author's VOS inference implementation on DAVIS 2017."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.vos_benchmarks import inventory, layout
from scripts.memory_token_io import digest, parameter_digest, validate_output


METHODS = {
    "samurai": {
        "repo": "external/SAMURAI",
        "revision": "76ba195984892b0d1e3db5d9c9f90bb62175680a",
        "config": "configs/samurai/sam2.1_hiera_t.yaml",
        "tool": "tools/vos_inference.py",
        "inference_function": "vos_inference",
        "precision_note": "CPU inference with FP32 model weights; the upstream CUDA autocast decorator is disabled on CPU.",
        "multi_object_note": "Exploratory system-level reference only: pinned SAMURAI source maintains one model-level Kalman state, so its motion cue is not independently tracked per DAVIS object.",
    },
    "sam2long": {
        "repo": "external/SAM2Long",
        "revision": "d70b50a7936fec55af201244ecde3d4433aff943",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "tool": "tools/vos_inference.py",
        "inference_function": "vos_separate_inference_per_object",
        "precision_note": "CPU inference with FP32 model weights; pinned SAM2Long stores memory features in BF16.",
        "multi_object_note": "The author per-object VOS wrapper is used; each object has its own inference state, with its pathway logic unchanged.",
        "vos_options": {"num_pathway": 3, "iou_thre": 0.1, "uncertainty": 2},
    },
}
PROTOCOL = ROOT / "protocols/davis_region_vs_temporal_selection_v1.json"
EVALUATOR_REVISION = "ac7c43fca936f9722837b7fbd337d284ba37004b"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_revision(repo):
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ):
        raise RuntimeError(f"Pinned author checkout is dirty: {repo}")
    return revision


def load_author_vos_tool(repo, relative):
    path = repo / relative
    spec = importlib.util.spec_from_file_location("_pinned_author_vos_tool", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load author VOS tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not any(callable(getattr(module, name, None)) for name in (
        "vos_inference", "vos_separate_inference_per_object"
    )):
        raise RuntimeError(f"Author VOS tool has no supported inference function: {path}")
    return module


def author_vos_function(method, module):
    """Select the exact entry point used by each repository's DAVIS VOS CLI."""
    name = METHODS[method]["inference_function"]
    function = getattr(module, name, None)
    if not callable(function):
        raise RuntimeError(f"{method} VOS entry point is unavailable: {name}")
    return function


def evaluator_summary(prediction_dir, root, records, split):
    evaluator_dir = ROOT / "external/davis2017-evaluation"
    revision = source_revision(evaluator_dir)
    if revision != EVALUATOR_REVISION:
        raise RuntimeError("Pinned DAVIS evaluator revision changed")
    if str(evaluator_dir) not in sys.path:
        sys.path.insert(0, str(evaluator_dir))
    from davis2017.evaluation import DAVISEvaluation
    from scripts.run_davis_soft_mask_aware_tome import summarize_official_metrics

    evaluator = DAVISEvaluation(
        davis_root=str(root), task="semi-supervised", gt_set=split,
        sequences=[record["video"] for record in records],
    )
    metrics = evaluator.evaluate(str(prediction_dir), metric=("J", "F"))
    return summarize_official_metrics(metrics, records)


def videos_for_stage(records, stage):
    """Select only the predeclared multi-object train video for the gate."""
    if stage == "gate":
        videos = [record for record in records if record["video"] == "bmx-bumps"]
        if len(videos) != 1 or videos[0]["object_ids"] != [1, 2]:
            raise RuntimeError("Two-object train engineering video changed")
        return videos
    if stage == "full":
        return list(records)
    raise ValueError(f"Unsupported DAVIS comparator stage: {stage}")


def validate_local_reference(local_run, dataset_root, checkpoint):
    """Reject incomplete or mismatched local results before system comparison."""
    setup_path = local_run / "setup.json"
    results_path = local_run / "results.json"
    inventory_path = local_run / "inventory.json"
    if not all(path.is_file() for path in (setup_path, results_path, inventory_path)):
        raise RuntimeError("Run 159 lacks setup, inventory, or final results")
    setup = json.loads(setup_path.read_text())
    results = json.loads(results_path.read_text())
    current_inventory = inventory(dataset_root, "davis2017", "val")
    saved_inventory = json.loads(inventory_path.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    required = set(protocol["shared_setup"]["local_reference_required_conditions"])
    if setup.get("dataset") != "davis2017" or setup.get("split") != "val":
        raise RuntimeError("Run 159 is not the DAVIS 2017 val reference")
    if setup.get("checkpoint_sha256") != digest(checkpoint):
        raise RuntimeError("Run 159 used a different checkpoint")
    if setup.get("model_config") != protocol["shared_setup"]["model_config"]:
        raise RuntimeError("Run 159 used a different SAM2 model configuration")
    if setup.get("indexed_mask_non_overlap_outputs") is not True:
        raise RuntimeError("Run 159 did not use the indexed-mask output policy")
    if setup.get("dataset_inventory_sha256") != digest(inventory_path):
        raise RuntimeError("Run 159 inventory receipt does not match its saved inventory")
    if saved_inventory != current_inventory:
        raise RuntimeError("Run 159 and author runs do not use the same DAVIS val inventory")
    conditions = results.get("conditions", {})
    if not required <= set(conditions):
        raise RuntimeError("Run 159 is missing one or more required local conditions")
    expected_videos = {record["video"] for record in current_inventory["videos"]}
    for condition in required:
        evaluation = conditions[condition].get("official_evaluation", {})
        if set(evaluation.get("per_video", {})) != expected_videos:
            raise RuntimeError(f"Run 159 has incomplete official results for {condition}")
    return {
        "setup": setup,
        "setup_sha256": digest(setup_path),
        "results_sha256": digest(results_path),
    }


def validate_prediction_folder(prediction_dir, video, record, root, split):
    _, mask_root, _, _ = layout(root, "davis2017", split)
    predicted = sorted(
        (prediction_dir / video).glob("*.png"), key=lambda path: int(path.stem)
    )
    expected_names = [Path(name).stem for name in record["frames"]]
    if [path.stem for path in predicted] != expected_names:
        raise RuntimeError(f"Incomplete or extra author masks for {video}")
    gt_prompt = np.asarray(Image.open(mask_root / video / f"{expected_names[0]}.png"))
    output_prompt = np.asarray(Image.open(predicted[0]))
    if output_prompt.shape != (record["height"], record["width"]):
        raise RuntimeError(f"Prompt output geometry changed for {video}")
    valid = gt_prompt != 255
    if not np.array_equal(output_prompt[valid], gt_prompt[valid]):
        raise RuntimeError(f"Author pipeline changed the supplied prompt mask for {video}")
    expected_ids = {0, *record["object_ids"]}
    for path in predicted:
        mask = np.asarray(Image.open(path))
        if mask.shape != (record["height"], record["width"]):
            raise RuntimeError(f"Prediction geometry changed at {video}/{path.name}")
        labels = set(int(value) for value in np.unique(mask))
        # The official Results reader treats every nonzero label as an object
        # index; 255 is reserved for void in annotations, not valid predictions.
        if not labels <= expected_ids:
            raise RuntimeError(f"Unexpected object IDs in {video}/{path.name}: {labels}")
    return {"frames": len(predicted), "prompt_mask_exact_on_valid_pixels": True}


def run_one_video(method, args, record, split, output_dir):
    spec = METHODS[method]
    repo = ROOT / spec["repo"]
    revision = source_revision(repo)
    if revision != spec["revision"]:
        raise RuntimeError(f"{method} checkout differs from its pinned revision")
    package_root = repo / "sam2" if method == "samurai" else repo
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    from sam2.build_sam import build_sam2_video_predictor

    model_config = spec["config"]
    config_path = package_root / "sam2" / model_config
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    started_load = time.monotonic()
    model = build_sam2_video_predictor(
        model_config,
        str(args.checkpoint),
        device="cpu",
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.non_overlap_masks=true"],
    )
    load_seconds = time.monotonic() - started_load
    model.eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(f"{method} did not build as a frozen eval model")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError(f"{method} model parameters are not CPU FP32")
    if not getattr(model, "non_overlap_masks", False):
        raise RuntimeError(f"{method} did not enable indexed-mask non-overlap behavior")
    if method == "samurai" and not getattr(model, "samurai_mode", False):
        raise RuntimeError("SAMURAI config did not activate SAMURAI mode")

    tool = load_author_vos_tool(package_root, spec["tool"])
    image_root, mask_root, _, _ = layout(args.dataset_root, "davis2017", split)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = parameter_digest(model)
    started = time.monotonic()
    kwargs = {
        "predictor": model,
        "base_video_dir": str(image_root),
        "input_mask_dir": str(mask_root),
        "output_mask_dir": str(output_dir),
        "video_name": record["video"],
        "score_thresh": 0.0,
        "use_all_masks": False,
        "per_obj_png_file": False,
    }
    kwargs.update(spec.get("vos_options", {}))
    author_vos_function(method, tool)(**kwargs)
    inference_seconds = time.monotonic() - started
    after = parameter_digest(model)
    if after != before:
        raise RuntimeError(f"{method} changed pretrained parameters on {record['video']}")
    prediction_check = validate_prediction_folder(
        output_dir, record["video"], record, args.dataset_root, split
    )
    return {
        "method": method,
        "video": record["video"],
        "split": split,
        "frames": prediction_check["frames"],
        "objects": record["object_ids"],
        "checkpoint_sha256": digest(args.checkpoint),
        "author_revision": revision,
        "model_config": model_config,
        "inference_function": spec["inference_function"],
        "implementation_note": spec["multi_object_note"],
        "device_precision": spec["precision_note"],
        "model_parameters_unchanged": True,
        "non_overlap_masks": True,
        "prompt_mask_exact_on_valid_pixels": prediction_check["prompt_mask_exact_on_valid_pixels"],
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "prediction_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((output_dir / record["video"]).glob("*.png"))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--stage", choices=("gate", "full"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-run-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.dataset_root = args.dataset_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    out = validate_output(args.output_dir)
    if not args.dataset_root.is_dir() or not args.checkpoint.is_file():
        parser.error("DAVIS root or SAM2.1 Tiny checkpoint is missing")
    protocol = json.loads(PROTOCOL.read_text())
    split = "train" if args.stage == "gate" else "val"
    inv = inventory(args.dataset_root, "davis2017", split)
    videos = videos_for_stage(inv["videos"], args.stage)
    local_reference = None
    if args.stage == "full":
        if args.local_run_dir is None:
            parser.error("--local-run-dir is required for the full comparison")
        local_run = args.local_run_dir.resolve()
        local_reference = validate_local_reference(
            local_run, args.dataset_root, args.checkpoint
        )

    out.mkdir(parents=True, exist_ok=True)
    setup = {
        "protocol_sha256": digest(PROTOCOL),
        "method": args.method,
        "stage": args.stage,
        "dataset": "davis2017",
        "split": split,
        "inventory_sha256": digest(args.dataset_root / f"ImageSets/2017/{split}.txt"),
        "checkpoint_sha256": digest(args.checkpoint),
        "author_revision": source_revision(ROOT / METHODS[args.method]["repo"]),
        "runner_sha256": digest(Path(__file__)),
        "model_config": METHODS[args.method]["config"],
        "non_overlap_masks": True,
        "model_overrides": ["++model.non_overlap_masks=true"],
        "apply_postprocessing": False,
        "inference_function": METHODS[args.method]["inference_function"],
        "implementation_note": METHODS[args.method]["multi_object_note"],
        "torch_num_threads": 2,
        "device_precision": METHODS[args.method]["precision_note"],
        "vos_options": METHODS[args.method].get("vos_options", {}),
    }
    if local_reference is not None:
        setup["local_reference_setup_sha256"] = local_reference["setup_sha256"]
        setup["local_reference_results_sha256"] = local_reference["results_sha256"]
    setup_path = out / f"{args.method}_{args.stage}_setup.json"
    if setup_path.is_file() and json.loads(setup_path.read_text()) != setup:
        raise RuntimeError("Comparator setup differs from its first recorded run")
    write_json(setup_path, setup)

    all_rows = []
    for index, record in enumerate(videos, start=1):
        method_dir = out / "conditions" / args.method
        predictions = method_dir / "predictions"
        receipt_path = method_dir / f"{record['video']}.complete.json"
        if args.stage == "full" and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("setup_sha256") != digest(setup_path):
                raise RuntimeError(f"Stale completion receipt: {args.method}/{record['video']}")
            all_rows.append(receipt)
            continue
        output_video_dir = predictions / record["video"]
        if output_video_dir.exists():
            import shutil
            shutil.rmtree(output_video_dir)
        result = run_one_video(args.method, args, record, split, predictions)
        result["setup_sha256"] = digest(setup_path)
        all_rows.append(result)
        write_json(out / "progress.json", {
            "stage": args.stage,
            "method": args.method,
            "video": record["video"],
            "videos_done": index,
            "videos_total": len(videos),
        })
        if args.stage == "full":
            write_json(receipt_path, result)
        print(f"[{args.method}] {args.stage} {record['video']} {index}/{len(videos)}", flush=True)

    if args.stage == "gate":
        evaluation = evaluator_summary(
            out / "conditions" / args.method / "predictions",
            args.dataset_root,
            videos,
            split,
        )
        write_json(out / f"{args.method}_gate.json", {
            "passed": True,
            "method": args.method,
            "video": videos[0]["video"],
            "scope": "Full DAVIS train/bmx-bumps engineering run; not an efficacy comparison.",
            "setup": setup,
            "result": all_rows[0],
            "official_evaluation_accepted": True,
            "official_evaluation": evaluation,
        })
        return

    val_inventory = inventory(args.dataset_root, "davis2017", "val")
    from scripts.run_davis_soft_mask_aware_tome import run_official_evaluator

    evaluation = run_official_evaluator(
        out / "conditions" / args.method / "predictions",
        args.dataset_root,
        val_inventory["videos"],
        split="val",
    )
    write_json(out / "conditions" / args.method / "official_evaluation.json", evaluation)
    write_json(out / f"{args.method}_full_summary.json", {
        "method": args.method,
        "setup": setup,
        "videos": len(all_rows),
        "total_model_load_seconds": sum(row["model_load_seconds"] for row in all_rows),
        "total_inference_seconds": sum(row["inference_seconds"] for row in all_rows),
        "official_evaluation": evaluation,
    })


if __name__ == "__main__":
    main()
