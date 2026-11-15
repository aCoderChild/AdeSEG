"""Real-frame engineering gate for ToMe with persistent DAVIS object IDs.

This checks object-bank routing only. It is not an accuracy evaluation.
The implementation reuses the project DAVIS prompt loader and SAM2 predictor.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.vos_benchmarks import author_mask_io, inventory, layout
from modeling.position_consistent_tome import PositionConsistentRegionToMe
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_davis_region_constrained_tome import frame_files, verify_prompt_inputs
from scripts.run_davis_soft_mask_aware_tome import build_model
from scripts.run_regional_refinement import write


PROTOCOL = ROOT / "protocols/position_consistent_region_tome_polypgen_v2.json"
DAVIS_ROOT = Path(
    "/Users/maianhpham/Library/CloudStorage/GoogleDrive-"
    "phammaianh11102005@gmail.com/My Drive/AdeSEG/datasets/DAVIS"
)
VIDEO = "bike-packing"
OBJECT_IDS = (1, 2)
FRAME_LIMIT = 3


def object_bank_map(state):
    result = {}
    for index, bank in state.get("output_dict_per_obj", {}).items():
        object_id = state.get("obj_idx_to_id", {}).get(index)
        if object_id is None:
            raise RuntimeError(f"No persistent ID for SAM2 object index {index}")
        result[id(bank)] = int(object_id)
    expected = sorted(int(value) for value in state.get("obj_ids", []))
    if not expected or sorted(result.values()) != expected:
        raise RuntimeError(f"Unexpected persistent object banks: {sorted(result.values())} != {expected}")
    return result


@torch.inference_mode()
def rollout(model, record, output_dir, order, *, use_tome):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root, _, _, _ = layout(DAVIS_ROOT, "davis2017", "val")
    state = model.init_state(
        video_path=str(image_root / VIDEO),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    images = frame_files(DAVIS_ROOT, VIDEO, "val")
    if len(images) != len(record["frames"]):
        raise RuntimeError("DAVIS frame inventory changed before prompt insertion")
    _, mask_root, _, _ = layout(DAVIS_ROOT, "davis2017", "val")
    source_labels, _ = author_mask_io()["load_ann_png"](
        str(mask_root / VIDEO / f"{images[0].stem}.png")
    )
    present_ids = sorted(int(value) for value in np.unique(source_labels) if value not in (0, 255))
    if not set(order).issubset(present_ids):
        raise RuntimeError(f"DAVIS prompt labels changed: {present_ids} do not contain {sorted(order)}")
    for object_id in order:
        prompt = torch.from_numpy((source_labels == object_id).copy())
        if not prompt.any():
            raise RuntimeError(f"Empty first-frame prompt for persistent ID {object_id}")
        model.add_new_mask(state, frame_idx=0, obj_id=object_id, mask=prompt)
    if list(state.get("obj_ids", [])) != list(order):
        raise RuntimeError("SAM2 did not retain the requested persistent-ID registration order")
    prompt_receipt = (
        verify_prompt_inputs(model, state, record, DAVIS_ROOT, "val")
        if sorted(order) == list(OBJECT_IDS) else None
    )
    mapping = object_bank_map(state)
    observer = (
        PositionConsistentRegionToMe(
            model, mode="regional", object_id_by_output_dict=mapping,
            provenance_target=1, hash_tensors_target=1,
        )
        if use_tome else None
    )
    iterator = None
    rows = {}
    start = time.monotonic()
    try:
        context = observer if observer is not None else _NullContext()
        with context:
            iterator = iter(model.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=FRAME_LIMIT - 1
            ))
            expected_frame = 0
            for frame_index, ids, logits in iterator:
                if frame_index != expected_frame:
                    raise RuntimeError(f"Out-of-order DAVIS frame: {frame_index}")
                if list(ids) != list(order):
                    raise RuntimeError(f"Persistent object order changed: {ids} != {order}")
                if logits.ndim != 4 or logits.shape[:2] != (len(order), 1):
                    raise RuntimeError(f"Invalid multi-object logits shape: {tuple(logits.shape)}")
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite DAVIS prediction logits")
                per_object = {}
                for axis, object_id in enumerate(order):
                    index = state["obj_id_to_idx"].get(object_id)
                    if index is None or state["obj_idx_to_id"].get(index) != object_id:
                        raise RuntimeError("SAM2 persistent object index mapping is not bijective")
                    bank = state["output_dict_per_obj"][index]
                    stored = bank["cond_frame_outputs"].get(frame_index)
                    if stored is None:
                        stored = bank["non_cond_frame_outputs"].get(frame_index)
                    if stored is None:
                        raise RuntimeError(f"Missing object-specific stored output {object_id}/{frame_index}")
                    for field in ("pred_masks", "maskmem_features", "object_score_logits"):
                        value = stored.get(field)
                        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                            raise RuntimeError(f"Invalid stored {field} for object {object_id}/{frame_index}")
                    per_object[str(object_id)] = {
                        "logits_sha256": tensor_hash(logits[axis:axis + 1]),
                        "stored_mask_sha256": tensor_hash(stored["pred_masks"]),
                        "stored_memory_sha256": tensor_hash(stored["maskmem_features"]),
                        "object_score_sha256": tensor_hash(stored["object_score_logits"]),
                    }
                rows[str(frame_index)] = per_object
                expected_frame += 1
            if expected_frame != FRAME_LIMIT:
                raise RuntimeError(f"Incomplete DAVIS ID gate: {expected_frame}/{FRAME_LIMIT} frames")
    finally:
        if iterator is not None:
            iterator.close()
        model.reset_state(state)

    calls = [] if observer is None else observer.calls
    if observer is not None:
        target_calls = [call for call in calls if call["frame"] > 0]
        expected = sorted(
            (frame, object_id, layer)
            for frame in range(1, FRAME_LIMIT)
            for object_id in order
            for layer in range(4)
        )
        observed = sorted(
            (int(call["frame"]), int(call["object_id"]), int(call["layer"]))
            for call in target_calls
        )
        if observed != expected:
            raise RuntimeError(f"Candidate did not make one four-layer read per object/frame: {observed}")
        for call in target_calls:
            object_id = int(call["object_id"])
            if object_id not in order:
                raise RuntimeError("Memory read was attributed to an unknown persistent ID")
            for frame in call["frames"]:
                if int(frame["t_pos"]) < 0:
                    raise RuntimeError("Missing SAM2 temporal slot for an object memory bank")
        # Capture one detailed membership trace, but avoid duplicating the full
        # source map on every target-frame read.
        write(output_dir / "cluster_provenance_first_target.json", observer.provenance_reads)

    return {
        "order": list(order),
        "used_tome": bool(use_tome),
        "frame_count": FRAME_LIMIT,
        "object_ids": list(order),
        "prompt_input_receipt": prompt_receipt,
        "wall_seconds": time.monotonic() - start,
        "frames": rows,
        "attention_reads": len(calls),
        "candidate_reads": len([call for call in calls if call["frame"] > 0]),
        "persistent_id_calls": sorted(set(int(call["object_id"]) for call in calls
                                           if call["object_id"] is not None)),
    }


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        protocol = json.loads(PROTOCOL.read_text())
        checkpoint_sha = digest(args.checkpoint)
        if checkpoint_sha != protocol["source_and_assets"]["sam2_checkpoint_sha256"]:
            raise RuntimeError("DAVIS ID gate checkpoint differs from frozen protocol")
        if not DAVIS_ROOT.is_dir():
            raise FileNotFoundError(DAVIS_ROOT)
        inv = inventory(DAVIS_ROOT, "davis2017", "val")
        record = next(item for item in inv["videos"] if item["video"] == VIDEO)
        if record["object_ids"] != list(OBJECT_IDS):
            raise RuntimeError("Frozen DAVIS multi-object IDs changed")
        if len(record["frames"]) < FRAME_LIMIT:
            raise RuntimeError("DAVIS bike-packing clip is too short for the ID gate")
        repo, revision = activate_author_repo()
        if revision != protocol["source_and_assets"]["sam2_revision"]:
            raise RuntimeError("SAM2 repository revision differs from the protocol")
        hashes = {
            name: digest(ROOT / name)
            for name in (
                "modeling/position_consistent_tome.py",
                "scripts/verify_position_consistent_tome_davis_ids.py",
                "scripts/run_davis_region_constrained_tome.py",
                "scripts/run_davis_soft_mask_aware_tome.py",
                "datasets/vos_benchmarks.py",
                "external/ToMe/tome/merge.py",
            )
        }
        write(output / "setup.json", {
            "protocol_sha256": digest(PROTOCOL),
            "checkpoint_sha256": checkpoint_sha,
            "davis_inventory_sha256": digest(DAVIS_ROOT / "ImageSets/2017/val.txt"),
            "sam2_revision": revision,
            "tome_source_sha256": digest(ROOT / "external/ToMe/tome/merge.py"),
            "source_hashes": hashes,
            "device": "cpu",
            "dtype": "float32",
            "video": VIDEO,
            "ids": list(OBJECT_IDS),
            "frames": FRAME_LIMIT,
        })
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        model = build_model(args.checkpoint)
        if len(model.memory_attention.layers) != 4 or model.training:
            raise RuntimeError("Expected frozen evaluation-mode four-layer SAM2")
        model.requires_grad_(False)
        initial_parameters = parameter_digest(model)
        results = {}
        for label, order, use_tome in (
            ("native", OBJECT_IDS, False),
            ("native_reverse", tuple(reversed(OBJECT_IDS)), False),
            ("region_tome", OBJECT_IDS, True),
            ("region_tome_reverse", tuple(reversed(OBJECT_IDS)), True),
            ("region_tome_id1_alone", (1,), True),
            ("region_tome_id2_alone", (2,), True),
        ):
            result = rollout(model, record, output / label, order, use_tome=use_tome)
            results[label] = result
            write(output / f"{label}.json", result)
            write(output / "progress.json", {"completed": label, "total": 6,
                                                "elapsed_seconds": time.monotonic() - started})

        for baseline_name, candidate_name in (
            ("native", "native_reverse"),
            ("region_tome", "region_tome_reverse"),
        ):
            a, b = results[baseline_name], results[candidate_name]
            for frame in range(FRAME_LIMIT):
                for object_id in OBJECT_IDS:
                    left = a["frames"][str(frame)][str(object_id)]
                    right = b["frames"][str(frame)][str(object_id)]
                    stored_fields = (
                        "stored_mask_sha256", "stored_memory_sha256", "object_score_sha256"
                    )
                    if any(left[key] != right[key] for key in stored_fields):
                        raise RuntimeError(
                            f"Registration order changed object {object_id}'s stored bank, frame {frame}"
                        )
                    # Frame-zero decoded logits are jointly consolidated with
                    # non-overlap rules and need not be prompt-order invariant.
                    if frame > 0 and left["logits_sha256"] != right["logits_sha256"]:
                        raise RuntimeError(
                            f"Registration order changed propagated output for object {object_id}, frame {frame}"
                        )
        multi = results["region_tome"]
        for object_id, single_name in ((1, "region_tome_id1_alone"), (2, "region_tome_id2_alone")):
            single = results[single_name]
            for frame in range(FRAME_LIMIT):
                multi_bank = multi["frames"][str(frame)][str(object_id)]
                single_bank = single["frames"][str(frame)][str(object_id)]
                if any(
                    multi_bank[key] != single_bank[key]
                    for key in ("stored_mask_sha256", "stored_memory_sha256", "object_score_sha256")
                ):
                    raise RuntimeError(
                        f"Other object's presence changed object {object_id}'s stored bank, frame {frame}"
                    )
        if parameter_digest(model) != initial_parameters:
            raise RuntimeError("Frozen SAM2 weights changed during DAVIS ID gate")
        if hashes != {
            name: digest(ROOT / name)
            for name in hashes
        }:
            raise RuntimeError("Source code changed during DAVIS ID gate")
        write(output / "completion.json", {
            "passed": True,
            "scope": "Engineering only: DAVIS val/bike-packing, persistent IDs 1 and 2, frames 0-2; no accuracy claim.",
            "registration_order_invariant": True,
            "per_object_stored_memory_isolated": True,
            "joint_non_overlap_logits_excluded_from_bank_isolation_claim": True,
            "all_four_layers_seen_per_object_target": True,
            "source_hashes_unchanged": True,
            "checkpoint_sha256": checkpoint_sha,
            "parameter_digest": initial_parameters,
            "results": {name: {key: value for key, value in row.items()
                               if key != "frames"} for name, row in results.items()},
            "elapsed_seconds": time.monotonic() - started,
        })
    except BaseException as error:
        write(output / "failure.json", {"passed": False, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
