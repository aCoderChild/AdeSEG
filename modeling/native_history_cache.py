"""Serialize native SAM2 banks and capture strictly past training reads."""

import tempfile
from pathlib import Path

import torch

from datasets.io import load_label_mask
from modeling.cfbi_training import capture_training_read
from modeling.native_history_training import block_positions, compact_history_read
from modeling.prediction_conditioned_affinity import memory_role_sequence
from scripts.bank_integrity import fingerprint
from scripts.run_author_baseline import tensor_hash
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_regional_refinement import load


def bank_payload(state, object_id=1):
    if object_id not in state["obj_id_to_idx"]:
        raise ValueError("Unknown persistent object ID")
    outputs = state["output_dict_per_obj"][state["obj_id_to_idx"][object_id]]
    if len(outputs["cond_frame_outputs"]) != 1:
        raise ValueError("Expected exactly one original prompt per cached object")
    prompt = next(iter(outputs["cond_frame_outputs"]))
    if set(outputs["non_cond_frame_outputs"]) != set(range(prompt + 1, state["num_frames"])):
        raise ValueError("Native bank does not cover the complete post-prompt video")
    return {"schema": 1, "object_id": object_id, "num_frames": state["num_frames"],
            "prompt": prompt, "outputs": outputs}


def load_bank(path, receipt, checkpoint_sha256, manifest_sha256, model_config):
    expected = {"checkpoint_sha256": checkpoint_sha256, "manifest_sha256": manifest_sha256,
                "model_config": model_config}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Native-bank provenance differs")
    payload = load(path, receipt["bank_sha256"])
    if payload.get("schema") != 1 or any(payload.get(key) != receipt[key]
            for key in ("object_id", "num_frames", "prompt")):
        raise ValueError("Native-bank metadata differs")
    if fingerprint(payload)["sha256"] != receipt["payload_sha256"]:
        raise RuntimeError("Serialized native bank changed values or layout")
    return payload


def cached_read(model, state, frame, payload):
    object_id = payload["object_id"]
    if (state["num_frames"] != payload["num_frames"] or object_id not in state["obj_id_to_idx"]
            or not payload["prompt"] < frame < payload["num_frames"]):
        raise ValueError("Cached read video, object or frame differs")
    index = state["obj_id_to_idx"][object_id]
    original = state["output_dict_per_obj"][index]
    state["output_dict_per_obj"][index] = {bucket: dict(entries)
                                          for bucket, entries in payload["outputs"].items()}
    try:
        model._get_image_feature(state, frame, 1)
        sample = capture_training_read(model, state, frame, object_id=object_id)
        stored = payload["outputs"]["non_cond_frame_outputs"][frame]
        with torch.inference_mode(False):
            sample["native_logits"] = stored["pred_masks"].detach().clone()
            sample["native_object_score"] = stored["object_score_logits"].detach().clone()
        if any(i >= frame for entries in sample["output_dict"].values() for i in entries):
            raise RuntimeError("Future memory entered a cached training read")
        return sample
    finally:
        state["output_dict_per_obj"][index] = original


def capture_cached_block(model, frames, selected_ids, output, payload):
    from sam2.modeling.position_encoding import PositionEmbeddingSine

    if payload["object_id"] != 1 or payload["num_frames"] != len(frames):
        raise ValueError("PolypGen block requires its single native object and full video")
    positions = block_positions(frames, selected_ids)
    before = fingerprint(payload)["sha256"]
    samples, records = [], []
    for module in model.modules():
        if isinstance(module, PositionEmbeddingSine):
            module.cache.clear()
    with tempfile.TemporaryDirectory(prefix="cached_history_", dir=output) as directory:
        with torch.inference_mode():
            state, prompt = prepare(model, frames, Path(directory))
            try:
                if prompt != payload["prompt"]:
                    raise RuntimeError("Cached and current original prompts differ")
                for frame in positions[1:]:
                    sample = cached_read(model, state, frame, payload)
                    with torch.inference_mode(False):
                        target = (load_label_mask(frames[frame].mask_path, [1]) == 1).copy()
                        sample["target"] = torch.from_numpy(target)[None, None].float()
                    _, spans = memory_role_sequence(model, frame, sample["output_dict"], sample["num_frames"])
                    packed = compact_history_read(sample, [s["frame"] for s in spans])
                    samples.append(packed)
                    records.append({"frame": frame, "spatial_frames": spans,
                                    "stored_mask_sha256": tensor_hash(sample["native_logits"])})
            finally:
                model.reset_state(state)
    if fingerprint(payload)["sha256"] != before:
        raise RuntimeError("Cached read or state reset changed the teacher bank")
    return samples, {"manifest_ids": list(selected_ids), "positions": positions,
        "num_frames": payload["num_frames"], "prompt": payload["prompt"],
        "payload_sha256": before, "source_bank_unchanged": True, "records": records}


def hybrid_memory_outputs(base, donor, *, component):
    """Diagnostic only: swap one stored component on identical past frame IDs.

    Spatial features and their positional encodings move together. Pointers
    move alone. Predicted masks/presence metadata stay with the base bank.
    The returned dictionaries share read-only tensors; no source is mutated.
    """
    fields = {"spatial": ("maskmem_features", "maskmem_pos_enc"),
              "pointer": ("obj_ptr",)}
    if component not in fields:
        raise ValueError("Expected spatial or pointer diagnostic component")
    result = {}
    for bucket, entries in base.items():
        result[bucket] = {}
        for frame, stored in entries.items():
            if frame not in donor.get(bucket, {}):
                raise ValueError("Donor lacks the same bucket/frame")
            other = donor[bucket][frame]
            changed = dict(stored)
            for field in fields[component]:
                if field not in stored or field not in other:
                    raise ValueError("Stored component is missing")
                changed[field] = other[field]
            result[bucket][frame] = changed
    return result
