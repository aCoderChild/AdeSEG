"""Gate then evaluate the frozen Run 155 method on DAVIS 2017 validation."""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.vos_benchmarks import author_mask_io, inventory, layout
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from modeling.davis_region_constrained_tome import (
    DavisObjectAwareRegionToMe,
    matched_plan_from_calls,
    validate_matched_calls,
)
from modeling.region_constrained_tome import TOME_REVISION
from modeling.region_tokens import SAM2_REVISION
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_davis_soft_mask_aware_tome import (
    EVALUATOR_REVISION,
    add_first_frame_prompts,
    build_model,
    run_official_evaluator,
    tensor_digest,
)


PROTOCOL = ROOT / "protocols/davis17_region_constrained_tome_v1.json"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
NATIVE_RUN_DEFAULT = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-"
    "phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/"
    "161_davis17_soft_mask_aware_tome_retry1_v1/full"
)
SOURCE_FILES = (
    "datasets/vos_benchmarks.py",
    "modeling/davis_region_constrained_tome.py",
    "modeling/prediction_conditioned_affinity.py",
    "modeling/region_constrained_tome.py",
    "modeling/region_tokens.py",
    "modeling/soft_role_affinity.py",
    "scripts/run_davis_region_constrained_tome.py",
    "scripts/run_davis_soft_mask_aware_tome.py",
    "tests/test_davis_region_constrained_tome.py",
    "tests/test_region_constrained_tome.py",
    "protocols/davis17_region_constrained_tome_v1.json",
    "external/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    "external/sam2/sam2/modeling/sam2_base.py",
    "external/sam2/sam2/sam2_video_predictor.py",
    "external/sam2/tools/vos_inference.py",
    "external/ToMe/tome/merge.py",
)
GATE_CONDITIONS = ("native", "identity_hooks", "region_constrained_tome", "generic_local_tome")
FULL_CONDITIONS = ("region_constrained_tome", "generic_local_tome")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_revision(repo, expected):
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != expected or dirty:
        raise RuntimeError(f"Pinned repository changed or is dirty: {repo} ({revision})")
    return revision


def hash_files(paths):
    return {name: digest(ROOT / name) for name in paths}


def frame_files(dataset_root, video, split):
    image_root, _, _, _ = layout(dataset_root, "davis2017", split)
    images = sorted(
        (path for path in (image_root / video).iterdir()
         if path.suffix.lower() in (".jpg", ".jpeg")),
        key=lambda path: int(path.stem),
    )
    if not images:
        raise RuntimeError(f"DAVIS video has no images: {split}/{video}")
    return images


def stored_object_hashes(state, frame, object_ids):
    index_to_id = state.get("obj_idx_to_id", {})
    result = {}
    for object_index, object_id in index_to_id.items():
        object_id = int(object_id)
        if object_id not in object_ids:
            raise RuntimeError(f"Unexpected per-object bank ID {object_id}")
        bank = state["output_dict_per_obj"][object_index]
        output = bank["cond_frame_outputs"].get(frame)
        if output is None:
            output = bank["non_cond_frame_outputs"].get(frame)
        if output is None:
            raise RuntimeError(f"Missing stored output for object {object_id}, frame {frame}")
        fields = {}
        for name in ("pred_masks", "maskmem_features", "object_score_logits"):
            value = output.get(name)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Stored object output is missing tensor {name}")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Nonfinite stored tensor {name} for object {object_id}/{frame}")
            fields[name] = tensor_digest(value)
        result[str(object_id)] = fields
    if sorted(int(value) for value in result) != sorted(object_ids):
        raise RuntimeError("Stored output hashes do not cover the prompted object IDs")
    return result


def verify_prompt_inputs(model, state, video_record, dataset_root, split):
    """Verify the exact SAM2 mask input after its native resize/threshold path.

    The decoded frame-zero prediction is not required to equal the prompt: it is
    still produced by SAM2's image encoder and mask decoder. The integrity check
    belongs on the stored prompt tensor itself.
    """
    _, mask_root, _, _ = layout(dataset_root, "davis2017", split)
    first_stem = Path(video_record["frames"][0]).stem
    labels, _ = author_mask_io()["load_ann_png"](
        str(mask_root / video_record["video"] / f"{first_stem}.png")
    )
    index_to_id = state.get("obj_idx_to_id", {})
    id_to_index = {int(object_id): int(index) for index, object_id in index_to_id.items()}
    object_ids = sorted(int(value) for value in video_record["object_ids"])
    if sorted(id_to_index) != object_ids:
        raise RuntimeError("Prompt insertion changed DAVIS persistent object IDs")

    receipts = {}
    for object_id in object_ids:
        prompt = torch.from_numpy((labels == object_id).copy())[None, None].float()
        prompt = prompt.to(state["device"])
        if tuple(prompt.shape[-2:]) != (int(model.image_size), int(model.image_size)):
            expected = torch.nn.functional.interpolate(
                prompt,
                size=(int(model.image_size), int(model.image_size)),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
            expected = (expected >= 0.5).float()
        else:
            expected = prompt
        try:
            actual = state["mask_inputs_per_obj"][id_to_index[object_id]][0]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"SAM2 did not retain the frame-zero mask prompt for object {object_id}") from exc
        if not torch.equal(actual, expected):
            raise RuntimeError(f"Stored SAM2 prompt tensor differs from source mask for object {object_id}")
        source_mask = np.ascontiguousarray(labels == object_id)
        receipts[str(object_id)] = {
            "source_object_id": object_id,
            "source_mask_sha256": hashlib.sha256(source_mask.tobytes()).hexdigest(),
            "sam2_input_tensor_sha256": tensor_digest(actual),
            "sam2_input_shape": list(actual.shape),
            "sam2_input_positive_pixels": int((actual >= 0.5).sum().item()),
            "matches_author_add_new_mask_resize": True,
        }
    return receipts


def verify_observer(observer, object_ids, frame_count, condition):
    if condition == "native":
        if observer is not None:
            raise RuntimeError("Native condition unexpectedly installed a memory hook")
        return {"reads": 0, "layers_per_object_target": 0}
    expected = sorted(
        (frame, object_id, layer)
        for frame in range(1, frame_count)
        for object_id in object_ids
        for layer in range(4)
    )
    observed = []
    pointer_reads = 0
    total_before = total_after = total_mass = total_keys = window_reads = 0
    for call in observer.calls:
        object_id = int(call["object_id"])
        frame = int(call["frame"])
        layer = int(call["layer"])
        observed.append((frame, object_id, layer))
        sources = call.get("mask_sources", [])
        if not sources or any(int(source["object_id"]) != object_id for source in sources):
            raise RuntimeError(f"Memory roles came from another object at {frame}/{object_id}")
        source_frames = [int(source["frame"]) for source in sources]
        if condition == "identity_hooks":
            if call["mode"] != "passthrough" or call["spatial_keys_after"] != call["spatial_keys_before"]:
                raise RuntimeError("Identity hook changed or compressed native memory")
            continue
        if [int(item["frame"]) for item in call["frames"]] != source_frames:
            raise RuntimeError(f"Mask provenance and key order differ at {frame}/{object_id}")
        if call["spatial_key_mass_sum"] != float(call["spatial_keys_before"]):
            raise RuntimeError("Spatial ToMe mass was not conserved")
        if call["key_mass_sum"] != float(call["spatial_keys_before"] + call["pointer_keys"]):
            raise RuntimeError("Pointer plus spatial attention mass was not conserved")
        if int(call["pointer_keys"]) > 0:
            pointer_reads += 1
        total_before += int(call["spatial_keys_before"])
        total_after += int(call["spatial_keys_after"])
        total_mass += float(call["spatial_key_mass_sum"])
        total_keys += float(call["key_mass_sum"])
        window_count = int(call.get("window_count", -1))
        if window_count < 1 or not call.get("window_budget_signature"):
            raise RuntimeError("A memory read lacks its compact local-window budget receipt")
        window_reads += window_count
        if call["spatial_keys_after"] >= call["spatial_keys_before"]:
            raise RuntimeError("ToMe did not reduce spatial memory keys")
    if sorted(observed) != expected:
        raise RuntimeError(f"Incomplete per-object/four-layer reads ({len(observed)} != {len(expected)})")
    summary = {"reads": len(observer.calls), "layers_per_object_target": 4}
    if condition != "identity_hooks":
        if pointer_reads == 0:
            raise RuntimeError("This DAVIS gate did not exercise any native object-pointer tokens")
        summary.update({
            "reads_with_pointer_tokens": pointer_reads,
            "spatial_keys_before": total_before,
            "spatial_keys_after": total_after,
            "spatial_key_mass_sum": total_mass,
            "total_key_mass_sum": total_keys,
            "window_reads": window_reads,
        })
    return summary


@torch.inference_mode()
def run_video_condition(condition, video_record, dataset_root, checkpoint, output_dir,
                        split, merge_plan=None, frame_limit=None):
    video = video_record["video"]
    object_ids = [int(value) for value in video_record["object_ids"]]
    images = frame_files(dataset_root, video, split)
    frame_count = len(images) if frame_limit is None else min(len(images), int(frame_limit))
    if frame_count < 2:
        raise ValueError("A DAVIS propagation gate needs a prompt and a target frame")
    model = build_model(checkpoint)
    before_parameters = parameter_digest(model)
    image_root, _, _, _ = layout(dataset_root, "davis2017", split)
    state = model.init_state(
        video_path=str(image_root / video),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    prompted_images, palette = add_first_frame_prompts(
        model, state, video, dataset_root, object_ids, split=split
    )
    prompt_receipt = verify_prompt_inputs(model, state, video_record, dataset_root, split)
    if [path.name for path in prompted_images] != [path.name for path in images]:
        raise RuntimeError("Prompt and propagation frame ordering differs")
    observer = None
    if condition == "identity_hooks":
        observer = DavisObjectAwareRegionToMe(model, state, mode="regional", passthrough=True)
    elif condition == "region_constrained_tome":
        observer = DavisObjectAwareRegionToMe(model, state, mode="regional")
    elif condition == "generic_local_tome":
        observer = DavisObjectAwareRegionToMe(
            model, state, mode="global", merge_plan_by_target=merge_plan
        )
    elif condition != "native":
        raise ValueError(f"Unknown DAVIS condition {condition}")

    mask_tools = author_mask_io()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, iterator = [], None
    start = time.monotonic()
    context = observer if observer is not None else _NullContext()
    try:
        with context as active_observer:
            iterator = iter(model.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=frame_count - 1
            ))
            expected_frame = 0
            for frame_index, output_ids, logits in iterator:
                if frame_index != expected_frame or output_ids != object_ids:
                    raise RuntimeError(
                        f"Unexpected frame or persistent object order {video}/{frame_index}: {output_ids}"
                    )
                if logits.ndim != 4 or logits.shape[:2] != (len(object_ids), 1):
                    raise RuntimeError(f"Unexpected multi-object logits shape {tuple(logits.shape)}")
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Nonfinite logits at {video}/{frame_index}")
                masks = {
                    object_id: (logits[index, 0] > 0).cpu().numpy()
                    for index, object_id in enumerate(output_ids)
                }
                mask_tools["save_masks_to_dir"](
                    str(output_dir), video, images[frame_index].stem, masks,
                    int(state["video_height"]), int(state["video_width"]),
                    False, palette or mask_tools["DAVIS_PALETTE"],
                )
                png_path = output_dir / video / f"{images[frame_index].stem}.png"
                with Image.open(png_path) as image:
                    indexed = np.asarray(image)
                if indexed.shape != (video_record["height"], video_record["width"]):
                    raise RuntimeError(f"Output PNG geometry changed at {video}/{frame_index}")
                row = {
                    "frame": int(frame_index),
                    "frame_name": images[frame_index].name,
                    "object_ids": list(output_ids),
                    "logits_sha256": tensor_digest(logits),
                    "prediction_png": f"{video}/{images[frame_index].stem}.png",
                    "prediction_png_sha256": digest(png_path),
                    "stored_object_outputs": stored_object_hashes(state, frame_index, object_ids),
                }
                records.append(row)
                expected_frame += 1
                if frame_index == frame_count - 1:
                    break
                if frame_index % 10 == 0:
                    write_json(Path(output_dir).parent / "progress.json", {
                        "stage": split,
                        "condition": condition,
                        "video": video,
                        "frames_done": expected_frame,
                        "frames_total": frame_count,
                    })
            if expected_frame != frame_count:
                raise RuntimeError(f"Incomplete video {video}: {expected_frame}/{frame_count} frames")
            observer_summary = verify_observer(
                active_observer, object_ids, frame_count, condition
            )
    finally:
        if iterator is not None:
            iterator.close()
        model.reset_state(state)
    after_parameters = parameter_digest(model)
    if before_parameters != after_parameters:
        raise RuntimeError(f"Frozen SAM2 weights changed during {condition}/{video}")
    if observer is not None:
        if condition == "identity_hooks":
            reference = getattr(run_video_condition, "_native_gate_rows", None)
            if reference is not None and records != reference:
                raise RuntimeError("Hooked pass-through differs from fresh native inference")
        if condition in ("region_constrained_tome", "generic_local_tome"):
            if len(observer.calls) != len(records[1:]) * len(object_ids) * 4:
                raise RuntimeError("Observed memory read count does not match frames, objects, and layers")
    return {
        "condition": condition,
        "split": split,
        "video": video,
        "frame_count": len(records),
        "object_ids": object_ids,
        "wall_seconds": time.monotonic() - start,
        "parameter_digest": before_parameters,
        "parameters_unchanged": True,
        "prompt_input_receipt": prompt_receipt,
        "observer": observer_summary,
        "frames": records,
        "calls": observer.calls if observer is not None else [],
    }


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def budget_signature(calls):
    rows = []
    for call in calls:
        window_count = int(call.get("window_count", -1))
        signature = call.get("window_budget_signature")
        if window_count < 1 or not signature:
            raise RuntimeError("Cannot summarize a memory read without its window receipt")
        rows.append((
            "layer", int(call["frame"]), int(call["object_id"]), int(call["layer"]),
            tuple(int(memory["frame"]) for memory in call["frames"]),
            window_count, signature,
        ))
        for memory in call["frames"]:
            for window in memory["windows"]:
                rows.append((
                    "window",
                    int(call["frame"]), int(call["object_id"]), int(call["layer"]),
                    int(memory["frame"]), int(memory["t_pos"]), int(window["window"]),
                    int(window["original_spatial_keys"]), int(window["merges"]),
                    int(window["retained_spatial_keys"]),
                ))
    rows.sort()
    encoded = json.dumps(rows, separators=(",", ":")).encode()
    return {
        "window_reads": sum(int(call["window_count"]) for call in calls),
        "layer_reads": len(calls),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def save_merge_plan(path, plan):
    rows = [
        {
            "target_frame": target,
            "object_id": object_id,
            "memory_frames": [
                {"frame": frame, "windows": {str(k): v for k, v in windows.items()}}
                for frame, windows in sorted(frames.items())
            ],
        }
        for (target, object_id), frames in sorted(plan.items())
    ]
    with gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0) as stream:
        stream.write(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())


def load_merge_plan(path):
    with gzip.open(path, "rt") as stream:
        rows = json.load(stream)
    return {
        (int(row["target_frame"]), int(row["object_id"])): {
            int(memory["frame"]): {int(k): int(v) for k, v in memory["windows"].items()}
            for memory in row["memory_frames"]
        }
        for row in rows
    }


def compact_result(result):
    result = dict(result)
    result.pop("calls", None)
    return result


def validate_native_reference(native_run, dataset_root, checkpoint, current_inventory):
    native_run = Path(native_run).resolve()
    setup_path = native_run / "setup.json"
    inventory_path = native_run / "inventory.json"
    results_path = native_run / "results.json"
    if not all(path.is_file() for path in (setup_path, inventory_path, results_path)):
        raise RuntimeError("Archived Run 161 lacks its setup, inventory, or results receipt")
    setup = json.loads(setup_path.read_text())
    saved_inventory = json.loads(inventory_path.read_text())
    results = json.loads(results_path.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    if setup_path.resolve() != (native_run / "setup.json").resolve():
        raise RuntimeError("Invalid Run 161 setup location")
    if digest(setup_path) != protocol["native_reference"]["setup_sha256"]:
        raise RuntimeError("Run 161 setup differs from the frozen native receipt")
    if digest(results_path) != protocol["native_reference"]["results_sha256"]:
        raise RuntimeError("Run 161 results differ from the frozen native receipt")
    if digest(inventory_path) != protocol["native_reference"]["inventory_sha256"]:
        raise RuntimeError("Run 161 inventory differs from the frozen native receipt")
    if setup.get("checkpoint_sha256") != digest(checkpoint):
        raise RuntimeError("Native DAVIS reference used a different checkpoint")
    if setup.get("dataset") != "davis2017" or setup.get("split") != "val":
        raise RuntimeError("Native reference is not DAVIS 2017 validation")
    if setup.get("model_config") != CONFIG or setup.get("sam2_revision") != SAM2_REVISION:
        raise RuntimeError("Native reference SAM2 configuration or revision differs")
    if setup.get("evaluator_revision") != EVALUATOR_REVISION:
        raise RuntimeError("Native reference evaluator revision differs")
    if setup.get("dataset_inventory_sha256") != digest(inventory_path):
        raise RuntimeError("Native reference inventory receipt hash is invalid")
    if setup.get("dataset_train_split_sha256") != digest(
        Path(dataset_root) / "ImageSets/2017/train.txt"
    ) or setup.get("dataset_val_split_sha256") != digest(
        Path(dataset_root) / "ImageSets/2017/val.txt"
    ):
        raise RuntimeError("Native reference uses different DAVIS split lists")
    if "++model.non_overlap_masks=true" not in setup.get("model_overrides", []):
        raise RuntimeError("Native reference did not use indexed non-overlap output handling")
    if setup.get("indexed_mask_non_overlap_outputs") is not True:
        raise RuntimeError("Native reference did not preserve indexed-object output semantics")
    if saved_inventory != current_inventory:
        raise RuntimeError("Native reference DAVIS validation inventory differs from current data")
    condition = results.get("conditions", {}).get("native", {})
    evaluation = condition.get("official_evaluation")
    if not evaluation or set(evaluation.get("per_video", {})) != {
        item["video"] for item in current_inventory["videos"]
    }:
        raise RuntimeError("Native reference lacks complete official per-video DAVIS metrics")

    _, mask_root, _, _ = layout(dataset_root, "davis2017", "val")
    setup_sha = digest(setup_path)
    for record in current_inventory["videos"]:
        video = record["video"]
        completion_path = native_run / "conditions/native" / f"{video}.complete.json"
        completion = json.loads(completion_path.read_text())
        if (completion.get("setup_sha256") != setup_sha
                or completion.get("condition") != "native"
                or completion.get("frame_count") != len(record["frames"])):
            raise RuntimeError(f"Native completion receipt is stale for {video}")
        video_result = condition.get(video)
        if not video_result or len(video_result.get("frames", [])) != len(record["frames"]):
            raise RuntimeError(f"Native result lacks full frame receipts for {video}")
        expected_frames = [Path(name).stem for name in record["frames"]]
        observed_frames = [int(row["frame"]) for row in video_result["frames"]]
        if observed_frames != list(range(len(expected_frames))):
            raise RuntimeError(f"Native reference frame sequence is incomplete for {video}")
        for row in video_result["frames"]:
            mask_path = native_run / "conditions/native/predictions" / row["prediction_png"]
            if not mask_path.is_file() or digest(mask_path) != row["prediction_png_sha256"]:
                raise RuntimeError(f"Native reference prediction failed its hash for {video}/{row['frame']}")
            if int(row["frame"]) == 0 and row["object_ids"] != record["object_ids"]:
                raise RuntimeError(f"Archived native prompt object IDs differ for {video}")
    return {
        "path": str(native_run),
        "setup_sha256": digest(setup_path),
        "results_sha256": digest(results_path),
        "inventory_sha256": digest(inventory_path),
        "official_evaluation": evaluation,
        "frame_zero_hashes": {
            video: {
                "logits_sha256": condition[video]["frames"][0]["logits_sha256"],
                "prediction_png_sha256": condition[video]["frames"][0]["prediction_png_sha256"],
            }
            for video in evaluation["per_video"]
        },
        "per_video_wall_seconds": {
            video: float(condition[video]["wall_seconds"])
            for video in evaluation["per_video"]
        },
    }


def source_receipt(checkpoint, dataset_root, train_inventory, val_inventory, native_reference):
    sam2 = ROOT / "external/sam2"
    tome = ROOT / "external/ToMe"
    evaluator = ROOT / "external/davis2017-evaluation"
    revisions = {
        "sam2": source_revision(sam2, SAM2_REVISION),
        "tome": source_revision(tome, TOME_REVISION),
        "davis_evaluator": source_revision(evaluator, EVALUATOR_REVISION),
    }
    return {
        "source_sha256": hash_files(SOURCE_FILES),
        "repository_revisions": revisions,
        "checkpoint_sha256": digest(checkpoint),
        "config_sha256": digest(sam2 / "sam2" / CONFIG),
        "dataset_root": str(Path(dataset_root).resolve()),
        "train_split_sha256": digest(Path(dataset_root) / "ImageSets/2017/train.txt"),
        "val_split_sha256": digest(Path(dataset_root) / "ImageSets/2017/val.txt"),
        "train_inventory_sha256": hashlib.sha256(
            json.dumps(train_inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "val_inventory_sha256": hashlib.sha256(
            json.dumps(val_inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "native_reference": {
            key: value for key, value in native_reference.items() if key != "official_evaluation"
        },
    }


def prepare_run_root(output_root, stage):
    root = validate_output(output_root)
    if stage == "gate":
        root.mkdir(parents=True, exist_ok=False)
        snapshot = root / "protocol_frozen.json"
        snapshot.write_bytes(PROTOCOL.read_bytes())
        write_json(root / "protocol_receipt.json", {
            "protocol_path": str(PROTOCOL),
            "protocol_sha256": digest(PROTOCOL),
            "snapshot_sha256": digest(snapshot),
            "git_head": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "working_tree_dirty_at_freeze": True,
        })
    else:
        snapshot = root / "protocol_frozen.json"
        receipt_path = root / "protocol_receipt.json"
        if not snapshot.is_file() or not receipt_path.is_file():
            raise RuntimeError("Full evaluation requires the frozen protocol snapshot")
        receipt = json.loads(receipt_path.read_text())
        if digest(PROTOCOL) != receipt["protocol_sha256"] or digest(snapshot) != receipt["snapshot_sha256"]:
            raise RuntimeError("DAVIS protocol changed after the engineering gate was frozen")
    return root


def load_setup(root, dataset_root, checkpoint, native_run):
    val_inventory = inventory(dataset_root, "davis2017", "val")
    train_inventory = inventory(dataset_root, "davis2017", "train")
    expected = json.loads(PROTOCOL.read_text())
    if (val_inventory["video_count"], val_inventory["object_count"], val_inventory["input_frames"]) != (30, 61, 1999):
        raise RuntimeError("DAVIS validation inventory differs from the frozen 30-video cohort")
    gate_video = next((row for row in train_inventory["videos"] if row["video"] == "bmx-bumps"), None)
    if gate_video is None or gate_video["object_ids"] != [1, 2] or len(gate_video["frames"]) != 90:
        raise RuntimeError("Frozen DAVIS two-object gate video changed")
    if digest(checkpoint) != expected["model"]["checkpoint_sha256"]:
        raise RuntimeError("SAM2 Tiny checkpoint differs from the frozen protocol")
    native_ref = validate_native_reference(native_run, dataset_root, checkpoint, val_inventory)
    setup = {
        "protocol_sha256": digest(PROTOCOL),
        "model_config": CONFIG,
        "evaluator_revision": EVALUATOR_REVISION,
        "native_reference": {key: value for key, value in native_ref.items() if key != "official_evaluation"},
        **source_receipt(checkpoint, dataset_root, train_inventory, val_inventory, native_ref),
    }
    write_json(root / "inventory_train.json", train_inventory)
    write_json(root / "inventory_val.json", val_inventory)
    return setup, train_inventory, val_inventory, native_ref


def save_video_receipt(path, result, setup_sha256):
    saved = compact_result(result)
    saved["setup_sha256"] = setup_sha256
    write_json(path, saved)


def load_plan_for_video(path):
    return load_merge_plan(path)


def run_gate(root, setup, train_inventory, dataset_root, checkpoint):
    setup_path = root / "setup.json"
    write_json(setup_path, setup)
    setup_sha = digest(setup_path)
    gate_root = root / "gate"
    video_record = next(row for row in train_inventory["videos"] if row["video"] == "bmx-bumps")
    condition_results = {}
    for condition in ("native", "identity_hooks"):
        folder = gate_root / "conditions" / condition / "predictions"
        result = run_video_condition(
            condition, video_record, dataset_root, checkpoint, folder, "train"
        )
        condition_results[condition] = result
        write_json(gate_root / "progress.json", {
            "condition": condition, "video": "bmx-bumps", "frames_done": result["frame_count"]
        })
    native_rows = condition_results["native"]["frames"]
    identity_rows = condition_results["identity_hooks"]["frames"]
    if native_rows != identity_rows:
        raise RuntimeError("Hooked identity path changed native masks, logits, or stored memory")

    regional_folder = gate_root / "conditions/region_constrained_tome/predictions"
    regional = run_video_condition(
        "region_constrained_tome", video_record, dataset_root, checkpoint,
        regional_folder, "train",
    )
    plan = matched_plan_from_calls(regional["calls"])
    generic_folder = gate_root / "conditions/generic_local_tome/predictions"
    generic = run_video_condition(
        "generic_local_tome", video_record, dataset_root, checkpoint,
        generic_folder, "train", merge_plan=plan,
    )
    matched = validate_matched_calls(regional["calls"], generic["calls"])
    for condition, result in (("region_constrained_tome", regional), ("generic_local_tome", generic)):
        if result["frame_count"] != len(video_record["frames"]):
            raise RuntimeError(f"Incomplete full-video engineering gate for {condition}")
        write_json(gate_root / "conditions" / condition / "run.json", compact_result(result))
    for condition in ("native", "identity_hooks"):
        write_json(gate_root / "conditions" / condition / "run.json",
                   compact_result(condition_results[condition]))

    record = run_official_evaluator(
        gate_root / "conditions/native/predictions", dataset_root,
        [video_record], split="train",
    )
    # The official evaluator accepts the gate folder for each method; this also
    # checks indexed-mask serialization for both region-aware conditions.
    evaluations = {"native": record}
    for condition in ("region_constrained_tome", "generic_local_tome"):
        evaluations[condition] = run_official_evaluator(
            gate_root / f"conditions/{condition}/predictions", dataset_root,
            [video_record], split="train",
        )
    budget = {
        "region_constrained_tome": budget_signature(regional["calls"]),
        "generic_local_tome": budget_signature(generic["calls"]),
    }
    if budget["region_constrained_tome"] != budget["generic_local_tome"]:
        raise RuntimeError("Two-object gate controls did not match exact local merge budgets")
    plan_path = gate_root / "matched_merge_plan.json.gz"
    save_merge_plan(plan_path, plan)
    gate = {
        "passed": True,
        "scope": "Full 90-frame DAVIS train/bmx-bumps two-object engineering gate; no validation efficacy result.",
        "setup_sha256": setup_sha,
        "video": "bmx-bumps",
        "object_ids": [1, 2],
        "frames": len(video_record["frames"]),
        "native_identity_exact": True,
        "identity_frame_hashes": len(native_rows),
        "region_observer": regional["observer"],
        "generic_observer": generic["observer"],
        "matched_budget": matched,
        "budget_signature": budget,
        "matched_merge_plan_sha256": digest(plan_path),
        "official_evaluation_accepted": True,
        "official_gate_scores": {
            key: value["global"] for key, value in evaluations.items()
        },
        "full_validation_started": False,
    }
    write_json(gate_root / "gate.json", gate)
    write_json(root / "status.json", {"state": "gate_passed", "gate_sha256": digest(gate_root / "gate.json")})
    return gate


def validate_completion(receipt_path, setup_sha, condition, record, prediction_dir):
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("setup_sha256") != setup_sha or receipt.get("condition") != condition
            or receipt.get("video") != record["video"]
            or receipt.get("frame_count") != len(record["frames"])):
        raise RuntimeError(f"Stale DAVIS video completion receipt: {receipt_path}")
    for frame in receipt["frames"]:
        path = prediction_dir / frame["prediction_png"]
        if not path.is_file() or digest(path) != frame["prediction_png_sha256"]:
            raise RuntimeError(f"Completed prediction missing or corrupt: {path}")
    return receipt


def dice_supplementary(prediction_dir, dataset_root, record):
    tools = author_mask_io()
    _, mask_root, _, _ = layout(dataset_root, "davis2017", "val")
    per_object = []
    frames = record["frames"]
    for index in range(1, len(frames) - 1):
        stem = Path(frames[index]).stem
        gt, _ = tools["load_ann_png"](str(mask_root / record["video"] / f"{stem}.png"))
        pred, _ = tools["load_ann_png"](str(prediction_dir / record["video"] / f"{stem}.png"))
        valid = gt != 255
        for object_id in record["object_ids"]:
            target = (gt == int(object_id)) & valid
            estimate = (pred == int(object_id)) & valid
            denominator = int(target.sum()) + int(estimate.sum())
            value = 1.0 if denominator == 0 else 2.0 * int((target & estimate).sum()) / denominator
            per_object.append(value)
    return float(np.mean(per_object)) if per_object else None


def summarize_contrasts(per_video):
    regional = per_video["region_constrained_tome"]
    native = per_video["native"]
    generic = per_video["generic_local_tome"]
    contrasts = {
        "region_constrained_tome - native": (regional, native),
        "region_constrained_tome - generic_local_tome": (regional, generic),
    }
    results, p_values = {}, {}
    for name, (left, right) in contrasts.items():
        common = sorted(set(left) & set(right))
        deltas = {video: left[video]["J&F-Mean"] - right[video]["J&F-Mean"] for video in common}
        test = paired_wilcoxon(
            {video: left[video]["J&F-Mean"] for video in common},
            {video: right[video]["J&F-Mean"] for video in common},
        )
        ci = sequence_bootstrap_interval(deltas, seed=20261106, resamples=10_000)
        wins = sum(value > 0 for value in deltas.values())
        ties = sum(value == 0 for value in deltas.values())
        results[name] = {
            "n_videos": len(common),
            "mean_delta": float(np.mean(list(deltas.values()))),
            "median_delta": float(np.median(list(deltas.values()))),
            "wins_ties_losses": [wins, ties, len(deltas) - wins - ties],
            "bootstrap_95_percentile_ci": [float(ci[0]), float(ci[1])],
            "wilcoxon_two_sided_p": float(test["p"]),
        }
        p_values[name] = float(test["p"])
    for name, value in holm_adjust(p_values).items():
        results[name]["holm_adjusted_p"] = float(value)
    return results


def make_final_results(native_ref, full_conditions):
    per_video = {"native": native_ref["official_evaluation"]["per_video"]}
    for condition in FULL_CONDITIONS:
        per_video[condition] = full_conditions[condition]["official_evaluation"]["per_video"]
    all_videos = sorted(per_video["native"])
    if any(set(per_video[key]) != set(all_videos) for key in per_video):
        raise RuntimeError("Official evaluation video coverage differs across conditions")
    contrasts = summarize_contrasts(per_video)
    equal_video_means = {
        condition: float(np.mean([per_video[condition][video]["J&F-Mean"] for video in all_videos]))
        for condition in per_video
    }
    dice_video = {"native": {}, "region_constrained_tome": {}, "generic_local_tome": {}}
    for condition in FULL_CONDITIONS:
        for video in all_videos:
            dice_video[condition][video] = full_conditions[condition]["per_video_dice"][video]
    for video in all_videos:
        native_video = full_conditions.get("native_per_video_dice", {})
        dice_video["native"][video] = native_video.get(video)
    equal_video_dice = {
        condition: float(np.mean([value for value in dice_video[condition].values() if value is not None]))
        for condition in dice_video
    }
    c_native = contrasts["region_constrained_tome - native"]
    c_generic = contrasts["region_constrained_tome - generic_local_tome"]
    screen = (
        c_native["mean_delta"] >= 0.01
        and c_generic["mean_delta"] > 0.0
        and c_native["bootstrap_95_percentile_ci"][0] > 0.0
        and c_generic["bootstrap_95_percentile_ci"][0] > 0.0
        and c_native["holm_adjusted_p"] < 0.05
        and c_generic["holm_adjusted_p"] < 0.05
        and c_native["wins_ties_losses"][0] > c_native["wins_ties_losses"][2]
        and c_generic["wins_ties_losses"][0] > c_generic["wins_ties_losses"][2]
    )
    return {
        "scope": "Exploratory DAVIS 2017 validation transfer comparison; validation informed earlier soft regional-ToMe development.",
        "primary_endpoint": "Official DAVIS semi-supervised J&F-Mean; video is the statistical unit.",
        "equal_video_mean_jf": equal_video_means,
        "equal_video_mean_dice_supplementary": equal_video_dice,
        "paired_video_contrasts": contrasts,
        "per_video_jf": per_video,
        "per_video_dice_supplementary": dice_video,
        "official_evaluation": {key: full_conditions[key]["official_evaluation"] for key in FULL_CONDITIONS},
        "meaningful_improvement_screen_passed": bool(screen),
        "screen": json.loads(PROTOCOL.read_text())["primary_evaluation"]["evidence_screen"],
    }


def write_results_markdown(path, results, full_conditions):
    means = results["equal_video_mean_jf"]
    dice = results["equal_video_mean_dice_supplementary"]
    order = ("native", "region_constrained_tome", "generic_local_tome")
    labels = {
        "native": "Native SAM2 (Run 161 archive)",
        "region_constrained_tome": "Run 155 region-constrained ToMe",
        "generic_local_tome": "Matched generic local ToMe",
    }
    lines = [
        "# Run 155 on DAVIS 2017 validation",
        "",
        "Exploratory transfer result. This benchmark was previously used for a different soft regional-ToMe variant.",
        "Official J&F-Mean is primary; Dice is supplementary. Means below weight videos equally.",
        "",
        "| Condition | Equal-video J&F | Equal-video Dice (supplementary) | Inference wall time (s) |",
        "|---|---:|---:|---:|",
    ]
    for condition in order:
        seconds = 0.0 if condition == "native" else sum(
            full_conditions[condition]["per_video_wall_seconds"].values()
        )
        lines.append(f"| {labels[condition]} | {means[condition]:.4f} | {dice[condition]:.4f} | {seconds:.1f} |")
    lines += [
        "",
        "## Paired video-level contrasts",
        "",
        "| Contrast | Mean Δ J&F | 95% paired video-bootstrap CI | Wins / ties / losses | Holm-adjusted p |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in results["paired_video_contrasts"].items():
        ci = item["bootstrap_95_percentile_ci"]
        w, t, l = item["wins_ties_losses"]
        lines.append(
            f"| {name} | {item['mean_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] "
            f"| {w} / {t} / {l} | {item['holm_adjusted_p']:.4g} |"
        )
    lines += [
        "",
        "## Per-video official J&F-Mean",
        "",
        "| Video | Native SAM2 | Region-constrained ToMe | Matched generic ToMe |",
        "|---|---:|---:|---:|",
    ]
    videos = sorted(results["per_video_jf"]["native"])
    for video in videos:
        values = [results["per_video_jf"][condition][video]["J&F-Mean"] for condition in order]
        lines.append(f"| {video} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += [
        "",
        f"Frozen meaningful-improvement screen: **{'PASS' if results['meaningful_improvement_screen_passed'] else 'FAIL'}**.",
        "",
        "The native arm was reused from Run 161 only after validating its complete DAVIS inventory, checkpoint, evaluator, per-video receipts, prompt masks, and prediction-file hashes. Published paper scores are not used as paired comparisons.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def run_full(root, setup, val_inventory, native_ref, dataset_root, checkpoint, gate_path, resume):
    gate = json.loads(Path(gate_path).read_text())
    if gate.get("passed") is not True or gate.get("setup_sha256") != digest(root / "setup.json"):
        raise RuntimeError("A passing matching two-object gate is required before DAVIS validation")
    setup_sha = digest(root / "setup.json")
    full_root = root / "full"
    if not full_root.exists():
        full_root.mkdir(parents=True)
    elif not resume:
        raise FileExistsError("DAVIS full output already exists; use --resume to continue it")
    conditions = {}
    for condition in FULL_CONDITIONS:
        condition_root = full_root / "conditions" / condition
        prediction_dir = condition_root / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        conditions[condition] = {
            "per_video_wall_seconds": {},
            "per_video_dice": {},
            "videos": {},
        }
        for record in val_inventory["videos"]:
            video = record["video"]
            receipt_path = condition_root / f"{video}.complete.json"
            plan_path = condition_root / f"{video}.merge_plan.json.gz"
            if condition == "generic_local_tome":
                regional_plan_path = (
                    full_root / "conditions/region_constrained_tome"
                    / f"{video}.merge_plan.json.gz"
                )
                if not regional_plan_path.is_file():
                    raise RuntimeError(f"Missing frozen regional budget for generic control {video}")
                regional_receipt_path = (
                    full_root / "conditions/region_constrained_tome"
                    / f"{video}.complete.json"
                )
                regional_receipt = json.loads(regional_receipt_path.read_text())
                if digest(regional_plan_path) != regional_receipt.get("merge_plan_sha256"):
                    raise RuntimeError(f"Frozen regional budget digest differs for {video}")
                plan = load_plan_for_video(regional_plan_path)
            else:
                plan = None
            if receipt_path.is_file():
                if not resume:
                    raise FileExistsError(f"Existing video receipt requires --resume: {receipt_path}")
                result = validate_completion(receipt_path, setup_sha, condition, record, prediction_dir)
            else:
                partial = prediction_dir / video
                if partial.exists():
                    shutil.rmtree(partial)
                result = run_video_condition(
                    condition, record, dataset_root, checkpoint,
                    prediction_dir, "val", merge_plan=plan,
                )
                if condition == "region_constrained_tome":
                    budget = matched_plan_from_calls(result["calls"])
                    save_merge_plan(plan_path, budget)
                    result["merge_plan_sha256"] = digest(plan_path)
                    result["budget_signature"] = budget_signature(result["calls"])
                else:
                    result["budget_signature"] = budget_signature(result["calls"])
                    region_receipt = json.loads(
                        (full_root / "conditions/region_constrained_tome" / f"{video}.complete.json").read_text()
                    )
                    if result["budget_signature"] != region_receipt.get("budget_signature"):
                        raise RuntimeError(f"Generic per-object token budget differs for {video}")
                if condition == "region_constrained_tome" and not plan_path.is_file():
                    raise RuntimeError(f"Regional merge plan was not saved for {video}")
                save_video_receipt(receipt_path, result, setup_sha)
                write_json(full_root / "progress.json", {
                    "condition": condition,
                    "video": video,
                    "videos_done": len(conditions[condition]["videos"]) + 1,
                    "videos_total": len(val_inventory["videos"]),
                })
            if condition == "generic_local_tome":
                regional_receipt = json.loads(
                    (full_root / "conditions/region_constrained_tome" / f"{video}.complete.json").read_text()
                )
                if result.get("budget_signature") != regional_receipt.get("budget_signature"):
                    raise RuntimeError(f"Resumed generic budget differs from regional plan for {video}")
            expected_prompt = native_ref["frame_zero_hashes"][video]
            actual_prompt = result["frames"][0]
            if any(actual_prompt.get(key) != value for key, value in expected_prompt.items()):
                raise RuntimeError(f"Frame-zero prediction differs from archived native SAM2 for {video}")
            result_record = compact_result(result)
            conditions[condition]["videos"][video] = result_record
            conditions[condition]["per_video_wall_seconds"][video] = float(result["wall_seconds"])
            conditions[condition]["per_video_dice"][video] = dice_supplementary(
                prediction_dir, dataset_root, record
            )

        conditions[condition]["official_evaluation"] = run_official_evaluator(
            prediction_dir, dataset_root, val_inventory["videos"], split="val"
        )
        write_json(condition_root / "official_evaluation.json", conditions[condition]["official_evaluation"])

    # Supplementary Dice for the archived native output uses the identical official scored frames.
    native_eval = native_ref["official_evaluation"]
    conditions["native_per_video_dice"] = {
        record["video"]: dice_supplementary(
            Path(setup["native_reference"]["path"]) / "conditions/native/predictions",
            dataset_root,
            record,
        )
        for record in val_inventory["videos"]
    }
    results = make_final_results(native_ref, {
        **conditions,
        "native_per_video_dice": conditions["native_per_video_dice"],
    })
    results["official_evaluation"]["native"] = native_eval
    results["setup_sha256"] = setup_sha
    write_json(full_root / "results.json", results)
    write_results_markdown(full_root / "results.md", results, {
        condition: {
            "per_video_wall_seconds": (
                conditions[condition]["per_video_wall_seconds"]
                if condition != "native" else native_ref["per_video_wall_seconds"]
            )
        }
        for condition in ("native", *FULL_CONDITIONS)
    })
    write_json(full_root / "completion.json", {
        "passed": True,
        "scope": "Complete 30-video DAVIS 2017 validation inference for both ToMe conditions; native reference verified from Run 161.",
        "setup_sha256": setup_sha,
        "conditions": ["native_reference", *FULL_CONDITIONS],
        "meaningful_improvement_screen_passed": results["meaningful_improvement_screen_passed"],
    })
    write_json(root / "status.json", {
        "state": "full_complete",
        "completion_sha256": digest(full_root / "completion.json"),
        "results_sha256": digest(full_root / "results.json"),
    })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gate", "full"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, default=NATIVE_RUN_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-receipt", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not dataset_root.is_dir() or not checkpoint.is_file():
        parser.error("DAVIS dataset or SAM2 Tiny checkpoint is unavailable")
    if args.stage == "full" and args.gate_receipt is None:
        parser.error("Full validation requires --gate-receipt")
    sam2_repo = ROOT / "external/sam2"
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    try:
        from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
    except ImportError as error:
        parser.error(f"Pinned SAM2 checkout is not importable: {error}")
    root = prepare_run_root(args.output_dir, args.stage)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        if args.stage == "gate":
            setup, train_inv, val_inv, native_ref = load_setup(
                root, dataset_root, checkpoint, args.native_run
            )
            run_gate(root, setup, train_inv, dataset_root, checkpoint)
        else:
            setup_path = root / "setup.json"
            gate_path = args.gate_receipt.resolve()
            if not setup_path.is_file():
                raise RuntimeError("Full validation has no engineering-gate setup receipt")
            setup = json.loads(setup_path.read_text())
            train_inv = json.loads((root / "inventory_train.json").read_text())
            val_inv = json.loads((root / "inventory_val.json").read_text())
            current_setup, current_train, current_val, native_ref = load_setup(
                root, dataset_root, checkpoint, args.native_run
            )
            if setup != current_setup or train_inv != current_train or val_inv != current_val:
                raise RuntimeError("Full DAVIS run provenance differs from the frozen gate")
            run_full(root, setup, val_inv, native_ref, dataset_root, checkpoint,
                     gate_path, args.resume)
    except BaseException as error:
        failure_dir = root / ("gate" if args.stage == "gate" else "full")
        failure_dir.mkdir(parents=True, exist_ok=True)
        write_json(failure_dir / "failure.json", {
            "stage": args.stage,
            "error": repr(error),
            "protocol_sha256": digest(root / "protocol_frozen.json")
            if (root / "protocol_frozen.json").is_file() else None,
        })
        write_json(root / "status.json", {"state": f"{args.stage}_failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
