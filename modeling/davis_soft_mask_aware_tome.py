"""Persistent-object adapter for the pinned soft-mask ToMe observer.

The underlying ToMe implementation remains unchanged. This adapter only keeps
each DAVIS object's writer-mask occupancy cache separate because SAM2's video
predictor invokes memory reads one object at a time.
"""

import torch
import torch.nn.functional as F

from modeling.soft_mask_aware_tome import NativeSoftMaskAwareToMe


class DavisObjectAwareToMe:
    """Route stored-mask occupancy by SAM2 persistent object ID."""

    def __init__(self, model, inference_state, mode="soft_mask_aware"):
        self.observer = NativeSoftMaskAwareToMe(model, mode=mode)
        self.inference_state = inference_state
        self.object_id_by_output_dict = {}
        self.mask_cache_by_object = {}
        self.active_object_id = None

        output_dicts = inference_state.get("output_dict_per_obj", {})
        index_to_id = inference_state.get("obj_idx_to_id", {})
        for object_index, output_dict in output_dicts.items():
            object_id = index_to_id.get(object_index)
            if object_id is None:
                raise RuntimeError(f"Missing persistent ID for object index {object_index}")
            self.object_id_by_output_dict[id(output_dict)] = int(object_id)
        if not self.object_id_by_output_dict:
            raise RuntimeError("DAVIS inference state contains no prompted objects")

        # Reuse the frozen observer's hooks and ToMe implementation, replacing
        # only its single-object bookkeeping with these per-object wrappers.
        self._base_capture = self.observer._capture_memory_write_mask
        self._base_track_step = self.observer._track_step_with_write_frame
        self._base_run_memory_encoder = self.observer._run_memory_encoder_with_write_frame
        self._base_prepare = self.observer._prepare
        self.observer._capture_memory_write_mask = self._capture_memory_write_mask
        self.observer._track_step_with_write_frame = self._track_step_with_write_frame
        self.observer._run_memory_encoder_with_write_frame = self._run_memory_encoder_with_write_frame
        self.observer._prepare = self._prepare

    def __enter__(self):
        self.observer.__enter__()
        return self.observer

    def __exit__(self, *exc):
        return self.observer.__exit__(*exc)

    def _object_id(self, output_dict):
        try:
            return self.object_id_by_output_dict[id(output_dict)]
        except KeyError as exc:
            raise RuntimeError("SAM2 used an unregistered per-object output dictionary") from exc

    @staticmethod
    def prompt_object_id(inference_state, frame, high_res_masks):
        """Match a preflight write to SAM2's exact low-res-to-high-res resize."""
        if not isinstance(high_res_masks, torch.Tensor) or high_res_masks.ndim != 4:
            raise ValueError("Expected SAM2 batched high-resolution mask logits")
        matches = []
        for object_index, temp_dict in inference_state.get("temp_output_dict_per_obj", {}).items():
            object_id = inference_state.get("obj_idx_to_id", {}).get(object_index)
            if object_id is None:
                continue
            for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                output = temp_dict[storage_key].get(frame)
                if output is None or output.get("maskmem_features") is not None:
                    continue
                low_res = output.get("pred_masks")
                if not isinstance(low_res, torch.Tensor):
                    continue
                expected = F.interpolate(
                    low_res.to(inference_state["device"]),
                    size=tuple(high_res_masks.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                )
                if torch.equal(expected, high_res_masks):
                    matches.append(int(object_id))
        if len(matches) != 1:
            raise RuntimeError(
                f"Prompt memory write at frame {frame} did not map to one persistent "
                f"object ID: {matches}"
            )
        return matches[0]

    def _capture_memory_write_mask(self, frame, mask_logits, memory_features, *, is_mask_from_pts=False):
        if self.active_object_id is None:
            raise RuntimeError("Memory-writer hook has no active persistent object ID")
        cache = self.mask_cache_by_object.setdefault(self.active_object_id, {})
        previous_cache = self.observer.mask_cache
        self.observer.mask_cache = cache
        try:
            self._base_capture(
                frame, mask_logits, memory_features, is_mask_from_pts=is_mask_from_pts
            )
        finally:
            self.observer.mask_cache = previous_cache
        cache[int(frame)][1]["object_id"] = int(self.active_object_id)

    def _track_step_with_write_frame(self, *args, **kwargs):
        bound = self.observer.track_step_signature.bind(*args, **kwargs)
        object_id = self._object_id(bound.arguments["output_dict"])
        previous = self.active_object_id
        self.active_object_id = object_id
        try:
            return self._base_track_step(*args, **kwargs)
        finally:
            self.active_object_id = previous

    def _run_memory_encoder_with_write_frame(self, *args, **kwargs):
        bound = self.observer.run_memory_encoder_signature.bind(*args, **kwargs)
        object_id = self.active_object_id
        if object_id is None:
            state = bound.arguments.get("inference_state")
            if state is None:
                raise RuntimeError("Cannot identify the object for a standalone memory write")
            object_id = self.prompt_object_id(
                state, int(bound.arguments["frame_idx"]), bound.arguments["high_res_masks"]
            )
        previous = self.active_object_id
        self.active_object_id = object_id
        try:
            return self._base_run_memory_encoder(*args, **kwargs)
        finally:
            self.active_object_id = previous

    def _prepare(self, *args, **kwargs):
        bound = self.observer.prepare_signature.bind(*args, **kwargs)
        object_id = self._object_id(bound.arguments["output_dict"])
        cache = self.mask_cache_by_object.get(object_id)
        if cache is None:
            raise RuntimeError(f"No memory-write occupancy cache exists for object {object_id}")
        previous_cache = self.observer.mask_cache
        self.observer.mask_cache = cache
        try:
            return self._base_prepare(*args, **kwargs)
        finally:
            self.observer.mask_cache = previous_cache
