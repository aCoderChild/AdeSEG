"""Run the frozen native/generic/soft-mask ToMe comparison on DAVIS17 val."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.vos_benchmarks import author_mask_io, inventory, layout
from modeling.davis_soft_mask_aware_tome import DavisObjectAwareToMe
from modeling.region_constrained_tome import TOME_REVISION
from modeling.soft_mask_aware_tome import MERGES_PER_WINDOW, ROLE_WEIGHT, WINDOW_SIZE
from modeling.region_tokens import SAM2_REVISION
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from scripts.memory_token_io import digest, parameter_digest, validate_output


PROTOCOL = ROOT / "protocols/davis_soft_mask_aware_tome_v1.json"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
EVALUATOR_REVISION = "ac7c43fca936f9722837b7fbd337d284ba37004b"
CONDITIONS = ("native", "generic_local_tome", "soft_mask_aware_tome")
SOURCE_FILES = (
    "datasets/vos_benchmarks.py",
    "modeling/davis_soft_mask_aware_tome.py",
    "modeling/soft_mask_aware_tome.py",
    "modeling/region_constrained_tome.py",
    "modeling/prediction_conditioned_affinity.py",
    "scripts/run_davis_soft_mask_aware_tome.py",
    "protocols/davis_soft_mask_aware_tome_v1.json",
    "external/sam2/sam2/sam2_video_predictor.py",
    "external/sam2/sam2/modeling/sam2_base.py",
    "external/sam2/tools/vos_inference.py",
)


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
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty:
        raise RuntimeError(f"Pinned author checkout is dirty: {repo}")
    return revision


def tensor_digest(tensor):
    value = tensor.detach().float().cpu().contiguous()
    if not torch.isfinite(value).all():
        raise RuntimeError("Prediction contains NaN or infinity")
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def build_model(checkpoint):
    from sam2.build_sam import build_sam2_video_predictor

    model = build_sam2_video_predictor(
        CONFIG,
        str(checkpoint),
        device="cpu",
        apply_postprocessing=False,
        # Match the official DAVIS indexed-mask inference path.
        hydra_overrides_extra=["++model.non_overlap_masks=true"],
    )
    # The upstream builder calls eval(), but leaves Parameter.requires_grad
    # enabled. This runner is inference-only: freeze parameters explicitly
    # without changing their values, then assert both inference invariants.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("DAVIS comparison requires frozen evaluation-mode SAM2")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("DAVIS comparison protocol requires CPU FP32 parameters")
    if len(model.memory_attention.layers) != 4:
        raise RuntimeError("Expected four native SAM2 memory-attention layers")
    if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1:
        raise RuntimeError("Native SAM2 memory schedule differs from the frozen protocol")
    if not model.non_overlap_masks:
        raise RuntimeError("DAVIS indexed-mask inference requires non-overlapping object outputs")
    return model


def video_files(root, video, split="val"):
    image_root, mask_root, _, per_object = layout(root, "davis2017", split)
    if per_object:
        raise RuntimeError("DAVIS indexed-mask layout unexpectedly changed")
    images = sorted(
        (path for path in (image_root / video).iterdir()
         if path.suffix.lower() in (".jpg", ".jpeg")),
        key=lambda path: int(path.stem),
    )
    return images, mask_root / video / f"{images[0].stem}.png", mask_root


def add_first_frame_prompts(model, state, video, root, object_ids, split="val"):
    images, prompt_path, _ = video_files(root, video, split=split)
    mask_tools = author_mask_io()
    labels, palette = mask_tools["load_ann_png"](str(prompt_path))
    if 255 in object_ids:
        raise RuntimeError("DAVIS void label was mistakenly listed as an object")
    present_ids = sorted(int(value) for value in np.unique(labels) if value > 0 and value != 255)
    if present_ids != sorted(object_ids):
        raise RuntimeError(f"First-frame IDs changed for {video}: {present_ids} != {object_ids}")
    for object_id in present_ids:
        prompt = torch.from_numpy((labels == object_id).copy())
        if not prompt.any():
            raise RuntimeError(f"Empty frame-0 prompt for {video}/{object_id}")
        model.add_new_mask(state, frame_idx=0, obj_id=object_id, mask=prompt)
    return images, palette


def window_budget_signature(calls):
    signature = []
    for call in calls:
        object_id = call["mask_sources"][0]["object_id"]
        for memory in call["frames"]:
            for window in memory["windows"]:
                signature.append((
                    int(call["frame"]), int(object_id), int(call["layer"]),
                    int(memory["frame"]), int(window["window"]),
                    int(window["original_spatial_keys"]),
                    int(window["merges"]), int(window["retained_spatial_keys"]),
                ))
    signature.sort()
    payload = json.dumps(signature, separators=(",", ":")).encode()
    return {"windows": len(signature), "sha256": hashlib.sha256(payload).hexdigest()}


def verify_observer_gate(observer, object_ids, frame_limit):
    if observer is None:
        return {"reads": 0, "checked_windows": 0, "budget_signature": None}
    expected = sorted((frame, object_id, layer)
                      for frame in range(1, frame_limit)
                      for object_id in object_ids for layer in range(4))
    observed = []
    checked_windows = 0
    for call in observer.calls:
        sources = call["mask_sources"]
        if not sources:
            raise RuntimeError("ToMe read has no stored-mask provenance")
        object_ids_in_sources = {int(source["object_id"]) for source in sources}
        if len(object_ids_in_sources) != 1:
            raise RuntimeError("Memory read mixed stored masks from different objects")
        object_id = object_ids_in_sources.pop()
        observed.append((int(call["frame"]), object_id, int(call["layer"])))
        if not all(source["object_id"] == object_id for source in sources):
            raise RuntimeError("Stored-mask occupancy was associated with the wrong object")
        if [source["frame"] for source in sources] != [item["frame"] for item in call["frames"]]:
            raise RuntimeError("Mask provenance and selected memory frames differ")
        if call["spatial_key_mass_sum"] != float(call["spatial_keys_before"]):
            raise RuntimeError("Spatial merge did not conserve attention mass")
        if call["key_mass_sum"] != float(call["spatial_keys_before"] + call["pointer_keys"]):
            raise RuntimeError("Total attention mass does not include unchanged pointers")
        if int(call["pointer_keys"]) <= 0:
            raise RuntimeError("Gate did not exercise SAM2 object-pointer preservation")
        if call["spatial_keys_after"] >= call["spatial_keys_before"]:
            raise RuntimeError("ToMe did not compress spatial memory keys")
        if any(source["memory_mask_binarized"] for source in sources):
            raise RuntimeError("DAVIS occupancy source was binarized before pooling")
        for source in sources:
            if source["memory_write_mask_shape"] != [1024, 1024]:
                raise RuntimeError("Unexpected memory-writer mask geometry")
            if source["occupancy_tokens"] != source["memory_grid_shape"][0] * source["memory_grid_shape"][1]:
                raise RuntimeError("Occupancy map does not cover the memory grid")
        if not call["frames"]:
            raise RuntimeError("ToMe read did not record memory-window budgets")
        for memory in call["frames"]:
            for window in memory["windows"]:
                if (
                    int(window["original_spatial_keys"]) != 16
                    or int(window["merges"]) != 8
                    or int(window["retained_spatial_keys"]) != 8
                ):
                    raise RuntimeError("A local memory window violated the frozen 16/8/8 budget")
                checked_windows += 1
    if sorted(observed) != expected:
        raise RuntimeError(f"Expected four reads per object/frame: {sorted(observed)} != {expected}")
    return {
        "reads": len(observer.calls),
        "checked_windows": checked_windows,
        "budget_signature": window_budget_signature(observer.calls),
    }


@torch.inference_mode()
def run_video_condition(condition, video, record, root, checkpoint, repo, output_dir,
                        frame_limit=None, split="val"):
    model = build_model(checkpoint)
    parameters_before = parameter_digest(model)
    image_root, _, _, _ = layout(root, "davis2017", split)
    images = sorted(
        (path for path in (image_root / video).iterdir()
         if path.suffix.lower() in (".jpg", ".jpeg")),
        key=lambda path: int(path.stem),
    )
    full_length = len(images)
    frame_limit = full_length if frame_limit is None else min(int(frame_limit), full_length)
    if frame_limit < 2:
        raise ValueError("DAVIS propagation requires frame 0 plus at least one target frame")

    state = model.init_state(
        video_path=str(image_root / video),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    images, palette = add_first_frame_prompts(
        model, state, video, root, record["object_ids"], split=split
    )
    if state["obj_ids"] != record["object_ids"]:
        raise RuntimeError(f"SAM2 persistent object IDs changed for {video}: {state['obj_ids']}")

    observer = None
    if condition == "generic_local_tome":
        observer = DavisObjectAwareToMe(model, state, mode="generic")
    elif condition == "soft_mask_aware_tome":
        observer = DavisObjectAwareToMe(model, state, mode="soft_mask_aware")

    out_dir = Path(output_dir)
    mask_tools = author_mask_io()
    records = []
    iterator = None
    started = time.monotonic()
    context = observer if observer is not None else nullcontext(None)
    try:
        with context as active_observer:
            iterator = iter(model.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=frame_limit - 1
            ))
            expected_frame = 0
            for frame_index, output_ids, logits in iterator:
                if frame_index != expected_frame or output_ids != record["object_ids"]:
                    raise RuntimeError(
                        f"Wrong DAVIS frame/object order {video}/{frame_index}: {output_ids}"
                    )
                if logits.ndim != 4 or logits.shape[:2] != (len(record["object_ids"]), 1):
                    raise RuntimeError(f"Unexpected multi-object output shape {tuple(logits.shape)}")
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Nonfinite output at {video}/{frame_index}")
                per_object = {
                    object_id: (logits[index, 0] > 0).cpu().numpy()
                    for index, object_id in enumerate(output_ids)
                }
                output_palette = palette or mask_tools["DAVIS_PALETTE"]
                mask_tools["save_masks_to_dir"](
                    str(out_dir), video, images[frame_index].stem, per_object,
                    int(state["video_height"]), int(state["video_width"]),
                    False, output_palette,
                )
                png_path = out_dir / video / f"{images[frame_index].stem}.png"
                png_sha256 = hashlib.sha256(png_path.read_bytes()).hexdigest()
                records.append({
                    "frame": int(frame_index),
                    "object_ids": list(output_ids),
                    "logits_sha256": tensor_digest(logits),
                    "prediction_png": f"{video}/{images[frame_index].stem}.png",
                    "prediction_png_sha256": png_sha256,
                })
                expected_frame += 1
                if frame_index == frame_limit - 1:
                    break
            if expected_frame != frame_limit:
                raise RuntimeError(f"Incomplete DAVIS trajectory for {video}: {expected_frame}/{frame_limit}")
            observer_gate = verify_observer_gate(
                active_observer, record["object_ids"], frame_limit
            )
    finally:
        if iterator is not None:
            iterator.close()
        model.reset_state(state)

    parameters_after = parameter_digest(model)
    if parameters_before != parameters_after:
        raise RuntimeError("Frozen SAM2 parameters changed during DAVIS propagation")
    return {
        "condition": condition,
        "split": split,
        "video": video,
        "frame_count": len(records),
        "object_ids": record["object_ids"],
        "wall_seconds": time.monotonic() - started,
        "parameters_unchanged": True,
        "parameter_digest": parameters_before,
        "observer": observer_gate,
        "frames": records,
    }


def source_receipt(repo, checkpoint, root):
    evaluator = ROOT / "external/davis2017-evaluation"
    revision = subprocess.check_output(
        ["git", "-C", str(evaluator), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EVALUATOR_REVISION:
        raise RuntimeError("Pinned DAVIS evaluator revision changed")
    if subprocess.check_output(
        ["git", "-C", str(evaluator), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ):
        raise RuntimeError("Pinned DAVIS evaluator checkout is dirty")
    tome = ROOT / "external/ToMe"
    tome_revision = subprocess.check_output(
        ["git", "-C", str(tome), "rev-parse", "HEAD"], text=True
    ).strip()
    if tome_revision != TOME_REVISION:
        raise RuntimeError("Pinned ToMe checkout revision changed")
    if subprocess.check_output(
        ["git", "-C", str(tome), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ):
        raise RuntimeError("Pinned ToMe checkout is dirty")
    return {
        "sam2_revision": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "evaluator_revision": revision,
        "model_overrides": ["++model.non_overlap_masks=true"],
        "indexed_mask_non_overlap_outputs": True,
        "checkpoint_sha256": digest(checkpoint),
        "dataset_train_split_sha256": digest(Path(root) / "ImageSets/2017/train.txt"),
        "dataset_val_split_sha256": digest(Path(root) / "ImageSets/2017/val.txt"),
        "tome_revision": tome_revision,
        "sources": {name: digest(ROOT / name) for name in SOURCE_FILES},
    }


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_official_metrics(metrics, video_records):
    """Keep official object-level outputs and derive the predeclared video unit."""
    j_metrics, f_metrics = metrics["J"], metrics["F"]
    per_object = []
    per_video = {}
    for record in video_records:
        video = record["video"]
        object_ids = sorted(int(value) for value in record["object_ids"])
        if object_ids != list(range(1, len(object_ids) + 1)):
            raise RuntimeError(f"DAVIS evaluator requires contiguous IDs for {video}: {object_ids}")
        video_rows = []
        for object_axis, object_id in enumerate(object_ids, start=1):
            key = f"{video}_{object_axis}"
            if key not in j_metrics["M_per_object"] or key not in f_metrics["M_per_object"]:
                raise RuntimeError(f"Official evaluator omitted {video}/object {object_id}")
            row = {
                "Sequence": key,
                "Video": video,
                "Object ID": object_id,
                "J-Mean": float(j_metrics["M_per_object"][key]),
                "F-Mean": float(f_metrics["M_per_object"][key]),
            }
            row["J&F-Mean"] = (row["J-Mean"] + row["F-Mean"]) / 2.0
            video_rows.append(row)
            per_object.append(row)
        if not video_rows:
            raise RuntimeError(f"No objects were scored for {video}")
        per_video[video] = {
            "J-Mean": float(np.mean([row["J-Mean"] for row in video_rows])),
            "F-Mean": float(np.mean([row["F-Mean"] for row in video_rows])),
            "J&F-Mean": float(np.mean([row["J&F-Mean"] for row in video_rows])),
            "objects": len(video_rows),
            "frames_evaluated": len(record["frames"]) - 2,
            "first_and_last_frames_excluded_by_official_protocol": True,
        }

    global_result = {
        "J&F-Mean": float((np.mean(j_metrics["M"]) + np.mean(f_metrics["M"])) / 2.0),
        "J-Mean": float(np.mean(j_metrics["M"])),
        "J-Recall": float(np.mean(j_metrics["R"])),
        "J-Decay": float(np.mean(j_metrics["D"])),
        "F-Mean": float(np.mean(f_metrics["M"])),
        "F-Recall": float(np.mean(f_metrics["R"])),
        "F-Decay": float(np.mean(f_metrics["D"])),
    }
    if len(per_object) != len(j_metrics["M"]) or len(per_object) != len(f_metrics["M"]):
        raise RuntimeError("Per-object records do not cover the official evaluator output")
    return {"global": global_result, "per_object": per_object, "per_video": per_video}


def summarize_local_contrasts(condition_results):
    """Paired comparison over 30 video means, not 61 objects or individual frames."""
    contrasts = {
        "soft_mask_aware_tome - native": ("soft_mask_aware_tome", "native"),
        "soft_mask_aware_tome - generic_local_tome": (
            "soft_mask_aware_tome", "generic_local_tome"
        ),
    }
    output = {}
    p_values = {}
    for name, (treatment, control) in contrasts.items():
        left = condition_results[treatment]["official_evaluation"]["per_video"]
        right = condition_results[control]["official_evaluation"]["per_video"]
        common = sorted(set(left) & set(right))
        differences = {
            video: left[video]["J&F-Mean"] - right[video]["J&F-Mean"]
            for video in common
        }
        test = paired_wilcoxon(
            {video: left[video]["J&F-Mean"] for video in common},
            {video: right[video]["J&F-Mean"] for video in common},
        )
        output[name] = {
            "n_videos": len(common),
            "mean_paired_delta": float(np.mean(list(differences.values()))),
            "median_paired_delta": float(np.median(list(differences.values()))),
            "wins_ties_losses": [
                sum(value > 0 for value in differences.values()),
                sum(value == 0 for value in differences.values()),
                sum(value < 0 for value in differences.values()),
            ],
            "bootstrap_95_percentile_ci": sequence_bootstrap_interval(
                differences, seed=20261105, resamples=10_000
            ),
            "wilcoxon_two_sided_p": test["p"],
        }
        p_values[name] = test["p"]
    adjusted = holm_adjust(p_values)
    for name, adjusted_p in adjusted.items():
        output[name]["holm_adjusted_p"] = adjusted_p
    return output


def run_official_evaluator(prediction_dir, root, video_records, split="val"):
    """Call the pinned official DAVIS API and retain its unrounded results."""
    evaluator_dir = ROOT / "external/davis2017-evaluation"
    if str(evaluator_dir) not in sys.path:
        sys.path.insert(0, str(evaluator_dir))
    from davis2017.evaluation import DAVISEvaluation

    evaluator = DAVISEvaluation(
        davis_root=str(root), task="semi-supervised", gt_set=split,
        sequences=[record["video"] for record in video_records],
    )
    metrics = evaluator.evaluate(str(prediction_dir), metric=("J", "F"))
    summary = summarize_official_metrics(metrics, video_records)
    global_csv = prediction_dir / f"global_results-{split}.csv"
    per_object_csv = prediction_dir / f"per-sequence_results-{split}.csv"
    _write_csv(global_csv, list(summary["global"]), [summary["global"]])
    _write_csv(
        per_object_csv,
        ["Sequence", "J-Mean", "F-Mean"],
        [
            {key: row[key] for key in ("Sequence", "J-Mean", "F-Mean")}
            for row in summary["per_object"]
        ],
    )
    _write_csv(
        prediction_dir / f"per-video_results-{split}.csv",
        ["Video", "Objects", "Frames Evaluated", "J-Mean", "F-Mean", "J&F-Mean"],
        [
            {
                "Video": video,
                "Objects": row["objects"],
                "Frames Evaluated": row["frames_evaluated"],
                "J-Mean": row["J-Mean"],
                "F-Mean": row["F-Mean"],
                "J&F-Mean": row["J&F-Mean"],
            }
            for video, row in summary["per_video"].items()
        ],
    )
    return {
        "official_evaluator_revision": EVALUATOR_REVISION,
        "official_api": "DAVISEvaluation.evaluate(metric=('J', 'F'))",
        "official_first_and_last_frame_excluded": True,
        **summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gate", "full"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-receipt", type=Path)
    args = parser.parse_args()

    out = validate_output(args.output_dir)
    root = args.dataset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not root.is_dir() or not checkpoint.is_file():
        parser.error("DAVIS dataset root or SAM2 checkpoint does not exist")
    if args.stage == "full" and args.gate_receipt is None:
        parser.error("--gate-receipt is required for full-cohort inference")
    if args.stage == "gate" and out.exists():
        parser.error("Gate output directory already exists; choose a fresh Drive directory")
    out.mkdir(parents=True, exist_ok=(args.stage == "full"))

    try:
        torch.set_num_threads(2)
        protocol = json.loads(PROTOCOL.read_text())
        if protocol["model"]["configuration"] != CONFIG:
            raise RuntimeError("Runner model configuration differs from the frozen DAVIS protocol")
        if protocol["evaluation"]["official_evaluator_revision"] != EVALUATOR_REVISION:
            raise RuntimeError("Runner evaluator revision differs from the frozen DAVIS protocol")
        if (protocol["method"]["window_size"], protocol["method"]["merges_per_window"],
                protocol["method"]["role_weight"]) != (WINDOW_SIZE, MERGES_PER_WINDOW, ROLE_WEIGHT):
            raise RuntimeError("Runner ToMe settings differ from the frozen DAVIS protocol")
        if [condition["id"] for condition in protocol["conditions"]] != list(CONDITIONS):
            raise RuntimeError("Runner conditions differ from the frozen DAVIS protocol")
        inv = inventory(root, "davis2017", "val")
        gate_inventory = inventory(root, "davis2017", "train")
        expected_counts = protocol["scope"]
        actual_counts = (inv["video_count"], inv["object_count"], inv["input_frames"])
        frozen_counts = (
            int(expected_counts["videos"]),
            int(expected_counts["objects"]),
            int(expected_counts["frames"]),
        )
        if actual_counts != frozen_counts:
            raise RuntimeError(f"DAVIS validation inventory changed: {actual_counts} != {frozen_counts}")
        gate_spec = protocol["engineering_gate"]
        if gate_spec["split"] != "train" or gate_spec["sequence"] != "bmx-bumps":
            raise RuntimeError("Runner gate split/video differs from the frozen DAVIS protocol")
        inv_path = out / "inventory.json"
        write_json(inv_path, inv)
        gate_inv_path = out / "gate_inventory.json"
        write_json(gate_inv_path, gate_inventory)
        repo = ROOT / "external/sam2"
        if source_revision(repo) != SAM2_REVISION:
            raise RuntimeError("Pinned SAM2 checkout revision changed")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        receipt = source_receipt(repo, checkpoint, root)
        setup = {
            "protocol_sha256": digest(PROTOCOL),
            "dataset": "davis2017",
            "split": "val",
            "frame_resolution": "480p",
            "dataset_inventory_sha256": digest(inv_path),
            "engineering_gate_inventory_sha256": digest(gate_inv_path),
            "model_config": CONFIG,
            **receipt,
        }
        if args.stage == "full":
            gate = json.loads(args.gate_receipt.read_text())
            if gate.get("passed") is not True or gate.get("setup") != setup:
                raise RuntimeError("DAVIS full run does not match the passing engineering gate")
        setup_path = out / "setup.json"
        if setup_path.is_file():
            if json.loads(setup_path.read_text()) != setup:
                raise RuntimeError("Resume setup differs from the original DAVIS run")
        else:
            write_json(setup_path, setup)
        setup_sha256 = digest(setup_path)

        if args.stage == "gate":
            record = next(item for item in gate_inventory["videos"] if item["video"] == "bmx-bumps")
            if record["object_ids"] != [1, 2]:
                raise RuntimeError("The selected DAVIS two-object gate video changed")
            condition_results = {}
            for condition in CONDITIONS:
                target = out / "conditions" / condition / "predictions"
                target.mkdir(parents=True)
                result = run_video_condition(
                    condition, record["video"], record, root, checkpoint, repo,
                    target, frame_limit=2, split="train",
                )
                condition_results[condition] = result
                write_json(out / "progress.json", {
                    "stage": "gate", "condition": condition, "video": record["video"],
                    "frames": result["frame_count"],
                })
            generic = condition_results["generic_local_tome"]["observer"]["budget_signature"]
            aware = condition_results["soft_mask_aware_tome"]["observer"]["budget_signature"]
            if generic != aware:
                raise RuntimeError("Generic and soft-mask ToMe use different per-window budgets")
            if any(result["frame_count"] != 2 for result in condition_results.values()):
                raise RuntimeError("Two-frame DAVIS gate coverage is incomplete")
            write_json(out / "gate.json", {
                "passed": True,
                "scope": "Engineering only: DAVIS 2017 train/bmx-bumps, two persistent objects, frames 0-1; no accuracy conclusion.",
                "setup": setup,
                "conditions": condition_results,
                "matched_window_budgets": generic["windows"],
                "full_cohort_inference_started": False,
            })
            return

        cohort = {record["video"]: record for record in inv["videos"]}
        results = {}
        for condition in CONDITIONS:
            condition_dir = out / "conditions" / condition
            prediction_dir = condition_dir / "predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            results[condition] = {}
            for video, record in cohort.items():
                receipt_path = condition_dir / f"{video}.complete.json"
                if receipt_path.is_file():
                    saved = json.loads(receipt_path.read_text())
                    if (
                        saved.get("setup_sha256") != setup_sha256
                        or saved.get("condition") != condition
                        or saved.get("video") != video
                        or saved.get("frame_count") != len(record["frames"])
                    ):
                        raise RuntimeError(f"Invalid or stale completion receipt for {condition}/{video}")
                    expected_masks = [
                        prediction_dir / video / f"{Path(name).stem}.png"
                        for name in record["frames"]
                    ]
                    if not all(path.is_file() for path in expected_masks):
                        raise RuntimeError(f"A completed DAVIS prediction is missing for {condition}/{video}")
                    results[condition][video] = saved
                    continue
                partial = prediction_dir / video
                if partial.exists():
                    shutil.rmtree(partial)
                result = run_video_condition(
                    condition, video, record, root, checkpoint, repo,
                    prediction_dir, frame_limit=None,
                )
                result["setup_sha256"] = setup_sha256
                write_json(receipt_path, result)
                results[condition][video] = result
                write_json(out / "progress.json", {
                    "stage": "full", "condition": condition, "video": video,
                    "videos_done": len(results[condition]), "videos_total": len(cohort),
                })
        for video in cohort:
            generic_budget = results["generic_local_tome"][video]["observer"]["budget_signature"]
            aware_budget = results["soft_mask_aware_tome"][video]["observer"]["budget_signature"]
            if generic_budget != aware_budget:
                raise RuntimeError(f"Generic and soft-mask budgets differ for {video}")
        for condition in CONDITIONS:
            condition_dir = out / "conditions" / condition
            prediction_dir = condition_dir / "predictions"
            results[condition]["official_evaluation"] = run_official_evaluator(
                prediction_dir, root, list(cohort.values())
            )
            write_json(condition_dir / "official_evaluation.json", results[condition]["official_evaluation"])
        local_contrasts = summarize_local_contrasts(results)
        write_json(out / "results.json", {
            "scope": "DAVIS 2017 val; official J/F/J&F; three local paired conditions.",
            "published_benchmark_numbers_are_context_only": True,
            "conditions": results,
            "local_video_level_contrasts": local_contrasts,
        })
    except BaseException as error:
        write_json(out / "failure.json", {"error": repr(error), "stage": args.stage})
        raise


if __name__ == "__main__":
    main()
