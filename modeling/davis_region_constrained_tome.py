"""Route the frozen region-ToMe observer through DAVIS object-specific banks."""

import hashlib
import json

from modeling.region_constrained_tome import NativeRegionConstrainedToMe


class DavisObjectAwareRegionToMe(NativeRegionConstrainedToMe):
    """Use SAM2's per-object banks while recording their persistent DAVIS IDs.

    The pinned video predictor invokes memory attention once per object, with a
    batch of one. This adapter keeps those calls separate and verifies that each
    role map was derived from the stored masks in that same object's bank.
    """

    def __init__(self, model, inference_state, mode="regional", *,
                 merge_plan_by_target=None, passthrough=False):
        object_id_by_output_dict = {}
        for object_index, output_dict in inference_state.get("output_dict_per_obj", {}).items():
            object_id = inference_state.get("obj_idx_to_id", {}).get(object_index)
            if object_id is None:
                raise RuntimeError(f"SAM2 has no persistent ID for object index {object_index}")
            object_id_by_output_dict[id(output_dict)] = int(object_id)
        expected_ids = sorted(int(value) for value in inference_state.get("obj_ids", []))
        mapped_ids = sorted(object_id_by_output_dict.values())
        if not mapped_ids or mapped_ids != expected_ids or len(set(mapped_ids)) != len(mapped_ids):
            raise RuntimeError(f"Per-object bank map is invalid: {mapped_ids} != {expected_ids}")
        super().__init__(
            model,
            mode=mode,
            merge_plan_by_target=merge_plan_by_target,
            object_id_by_output_dict=object_id_by_output_dict,
            passthrough=passthrough,
        )
        self.object_id_by_output_dict = object_id_by_output_dict

    def _prepare(self, *args, **kwargs):
        bound = self.prepare_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        data = bound.arguments
        try:
            object_id = self.object_id_by_output_dict[id(data["output_dict"])]
        except KeyError as exc:
            raise RuntimeError("DAVIS read used a bank without a persistent object ID") from exc
        call_start = len(self.calls)
        result = super()._prepare(*args, **kwargs)
        if data["is_init_cond_frame"]:
            return result

        outputs = {
            int(frame): output
            for frame, output in data["output_dict"]["non_cond_frame_outputs"].items()
        }
        outputs.update({
            int(frame): output
            for frame, output in data["output_dict"]["cond_frame_outputs"].items()
        })
        sources = []
        for span in self.spans:
            frame = int(span["frame"])
            output = outputs.get(frame)
            if output is None:
                raise RuntimeError(f"Object {object_id} has no stored output for frame {frame}")
            mask = output.get("pred_masks")
            features = output.get("maskmem_features")
            if (mask is None or mask.ndim != 4 or mask.shape[:2] != (1, 1)
                    or tuple(mask.shape[-2:]) != (256, 256)):
                raise RuntimeError(f"Unexpected stored-mask shape for object {object_id}/{frame}")
            if features is None or features.ndim != 4 or features.shape[0] != 1:
                raise RuntimeError(f"Unexpected memory-feature batch for object {object_id}/{frame}")
            sources.append({
                "object_id": object_id,
                "frame": frame,
                "t_pos": int(span["t_pos"]),
                "stored_pred_mask_sha256": _tensor_sha256(mask),
                "stored_pred_mask_shape": list(mask.shape),
                "memory_grid_shape": list(features.shape[-2:]),
            })

        frame = int(data["frame_idx"])
        for call in self.calls[call_start:]:
            if call.get("object_id") != object_id or int(call["frame"]) != frame:
                raise RuntimeError("A memory read was attributed to the wrong DAVIS object/frame")
            call["mask_sources"] = sources
            if [item["frame"] for item in sources] != [item["frame"] for item in call.get("frames", sources)]:
                raise RuntimeError("Stored-mask provenance differs from the selected memory frames")
        if len(self.calls) == call_start:
            raise RuntimeError(f"No memory-attention reads were observed for {object_id}/{frame}")
        if not self.passthrough:
            read_calls = self.calls[call_start:]
            signatures = []
            for call in read_calls:
                rows = [
                    (
                        int(memory["frame"]), int(memory["t_pos"]), int(window["window"]),
                        int(window["original_spatial_keys"]), int(window["merges"]),
                        int(window["retained_spatial_keys"]),
                    )
                    for memory in call["frames"]
                    for window in memory["windows"]
                ]
                if not rows:
                    raise RuntimeError("A memory layer has no local ToMe window audit")
                signature = hashlib.sha256(
                    json.dumps(rows, separators=(",", ":")).encode()
                ).hexdigest()
                call["window_budget_signature"] = signature
                call["window_count"] = len(rows)
                signatures.append((int(call["layer"]), signature))
            if sorted(layer for layer, _ in signatures) != [0, 1, 2, 3]:
                raise RuntimeError("DAVIS memory read did not visit all four attention layers")
            if len({signature for _, signature in signatures}) != 1:
                raise RuntimeError("Per-layer memory-key budgets differ for one object/frame")
            # Keep detailed window records once per object/frame. Other layers
            # retain the hash that proves they used the identical budget.
            for call in read_calls:
                if int(call["layer"]) != 0:
                    for memory in call["frames"]:
                        memory["windows"] = []
        return result


def _tensor_sha256(tensor):
    import hashlib

    value = tensor.detach().float().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def matched_plan_from_calls(calls):
    """Freeze the regional window budget for each target and persistent object."""
    grouped = {}
    for call in calls:
        key = (int(call["frame"]), int(call["object_id"]))
        grouped.setdefault(key, {}).setdefault(int(call["layer"]), call)
    plan = {}
    for key, layers in grouped.items():
        if sorted(layers) != [0, 1, 2, 3]:
            raise RuntimeError(f"Cannot freeze incomplete layer reads at {key}")
        reference = layers[0]["frames"]
        if any(not memory["windows"] for memory in reference):
            raise RuntimeError(f"Layer 0 lacks detailed window budgets at {key}")
        frame_plan = {}
        for memory in reference:
            frame_id = int(memory["frame"])
            if frame_id in frame_plan:
                raise RuntimeError(f"Duplicate selected memory frame {frame_id} at {key}")
            windows = memory["windows"]
            window_ids = [int(window["window"]) for window in windows]
            if window_ids != list(range(len(window_ids))):
                raise RuntimeError(f"Incomplete or reordered windows at {key}/{frame_id}")
            frame_plan[frame_id] = {
                int(window["window"]): int(window["merges"])
                for window in windows
            }
        for layer, call in layers.items():
            if [int(memory["frame"]) for memory in call["frames"]] != list(frame_plan):
                raise RuntimeError(f"Regional layer memory frames differ at {key}, layer {layer}")
            signature = call.get("window_budget_signature")
            if signature is None or signature != layers[0].get("window_budget_signature"):
                raise RuntimeError(f"Regional layer budgets differ at {key}, layer {layer}")
            if int(call.get("window_count", -1)) != sum(map(len, frame_plan.values())):
                raise RuntimeError(f"Regional layer window count differs at {key}, layer {layer}")
        plan[key] = frame_plan
    if not plan:
        raise RuntimeError("Regional run produced no merge plans")
    return plan


def validate_matched_calls(regional_calls, generic_calls):
    """Require identical per-object/per-target/per-layer/window merge counts."""
    def signature(calls):
        rows = []
        for call in calls:
            if call.get("window_budget_signature") is None or int(call.get("window_count", -1)) < 1:
                raise RuntimeError("A memory layer is missing its compact window-budget receipt")
            rows.append((
                "layer", int(call["frame"]), int(call["object_id"]), int(call["layer"]),
                tuple(int(memory["frame"]) for memory in call["frames"]),
                int(call["window_count"]), call["window_budget_signature"],
            ))
            for memory in call["frames"]:
                for window in memory["windows"]:
                    rows.append((
                        "window", int(call["frame"]), int(call["object_id"]), int(call["layer"]),
                        int(memory["frame"]), int(memory["t_pos"]), int(window["window"]),
                        int(window["original_spatial_keys"]), int(window["merges"]),
                        int(window["retained_spatial_keys"]),
                    ))
        return sorted(rows)

    regional, generic = signature(regional_calls), signature(generic_calls)
    if regional != generic:
        raise RuntimeError("Region and generic ToMe controls have unmatched token budgets")
    layer_keys = [
        (int(call["frame"]), int(call["object_id"])) for call in regional_calls
    ]
    expected_layers = {
        (target, object_id, layer)
        for target, object_id in layer_keys
        for layer in range(4)
    }
    observed_layers = {
        (int(call["frame"]), int(call["object_id"]), int(call["layer"]))
        for call in regional_calls
    }
    if observed_layers != expected_layers or len(regional_calls) != len(observed_layers):
        raise RuntimeError("Matched budget receipts do not cover exactly four layers per object/frame")
    return {
        "matched_window_reads": sum(int(call.get("window_count", 0)) for call in regional_calls),
        "exact_match": True,
    }
