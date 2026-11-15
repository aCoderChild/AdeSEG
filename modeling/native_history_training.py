"""Capture training blocks from the unchanged SAM2 video predictor."""

import tempfile
from pathlib import Path

import torch

from datasets.io import load_label_mask
from modeling.cfbi_compact_snapshot import compact_training_read
from modeling.cfbi_training import capture_training_read
from modeling.prediction_conditioned_affinity import memory_role_sequence
from scripts.bank_integrity import fingerprint
from scripts.run_author_baseline import tensor_hash
from scripts.run_cfbi_region_pilot import prepare


def block_positions(frames, selected_ids):
    ids = [frame.frame_index for frame in frames]
    if len(set(ids)) != len(ids) or len(selected_ids) < 3:
        raise ValueError("Expected unique manifest IDs and at least three block frames")
    lookup = {value: index for index, value in enumerate(ids)}
    if any(value not in lookup for value in selected_ids):
        raise ValueError("Block contains a frame outside this video")
    positions = [lookup[value] for value in selected_ids]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise ValueError("Expected consecutive manifest positions")
    return positions


def compact_history_read(sample, spatial_frames):
    packed = compact_training_read(sample, spatial_frames)
    def pointers(outputs):
        return {bucket: {frame: {"obj_ptr": value["obj_ptr"],
            "object_score_logits": value.get("object_score_logits")}
            for frame, value in entries.items()} for bucket, entries in outputs.items()}
    if fingerprint(pointers(sample["output_dict"]))["sha256"] != fingerprint(pointers(packed["output_dict"]))["sha256"]:
        raise RuntimeError("Compaction changed pointer inputs")
    for bucket, entries in sample["output_dict"].items():
        for frame in set(entries) & set(spatial_frames):
            if fingerprint(entries[frame])["sha256"] != fingerprint(packed["output_dict"][bucket][frame])["sha256"]:
                raise RuntimeError("Compaction changed selected spatial inputs")
    return packed


def capture_history_block(model, frames, selected_ids, output, reference):
    from sam2.modeling.position_encoding import PositionEmbeddingSine

    positions = block_positions(frames, selected_ids)
    targets = set(positions[1:])
    reference = {row["frame"]: row for row in reference}
    for module in model.modules():
        if isinstance(module, PositionEmbeddingSine):
            module.cache.clear()
    full, compact, records = [], [], []
    prefix_checks = 0
    with tempfile.TemporaryDirectory(prefix="native_history_", dir=output) as directory:
        with torch.inference_mode():
            state, prompt = prepare(model, frames, Path(directory))
            try:
                if positions[1] <= prompt or state["num_frames"] != len(frames):
                    raise ValueError("Targets must follow the original video prompt")
                for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                    if ids != [1] or not torch.isfinite(logits).all():
                        raise RuntimeError("Invalid native instance or prediction")
                    if frame > prompt:
                        if tensor_hash(logits) != reference[frame]["prediction_sha256"]:
                            raise RuntimeError(f"Native history replay differs at frame {frame}")
                        prefix_checks += 1
                    if frame in targets:
                        bank = state["output_dict_per_obj"][state["obj_id_to_idx"][1]]
                        before = fingerprint(bank)["sha256"]
                        sample = capture_training_read(model, state, frame, object_id=1)
                        stored = bank["non_cond_frame_outputs"][frame]
                        with torch.inference_mode(False):
                            sample["native_logits"] = stored["pred_masks"].detach().clone()
                            sample["native_object_score"] = stored["object_score_logits"].detach().clone()
                            target = (load_label_mask(frames[frame].mask_path, [1]) == 1).copy()
                            sample["target"] = torch.from_numpy(target)[None, None].float()
                        _, spans = memory_role_sequence(model, frame, sample["output_dict"], sample["num_frames"])
                        source_sha = fingerprint(sample["output_dict"])["sha256"]
                        packed = compact_history_read(sample, [span["frame"] for span in spans])
                        if fingerprint(bank)["sha256"] != before or fingerprint(sample["output_dict"])["sha256"] != source_sha:
                            raise RuntimeError("Capture or compaction mutated the source bank")
                        full.append(sample)
                        compact.append(packed)
                        records.append({"frame": frame, "manifest_id": frames[frame].frame_index,
                            "spatial_frames": spans, "native_prediction_sha256": tensor_hash(logits),
                            "stored_mask_sha256": tensor_hash(stored["pred_masks"]),
                            "source_bank_sha256": before, "source_bank_unchanged": True,
                            "pointer_inputs_identical": True, "spatial_inputs_identical": True})
                    if frame == positions[-1]:
                        break
            finally:
                model.reset_state(state)
    if [sample["frame"] for sample in full] != positions[1:]:
        raise RuntimeError("Incomplete native-history training block")
    return full, compact, {"manifest_ids": list(selected_ids), "positions": positions,
        "prompt": prompt, "num_frames": len(frames), "native_prefix_checks": prefix_checks,
        "pointer_time_denominator": min(len(frames), model.max_obj_ptrs_in_encoder) - 1,
        "records": records}
