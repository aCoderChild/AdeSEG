"""Bounded selection over MedSAM2's native explicit memory entries.

This predictor does not create, project, or fuse memory features.  It retains
the prompted conditioning frame (the anchor) and a fixed number of native
non-conditioning outputs.  The upstream MedSAM2 memory-attention code then
uses those unmodified outputs, including their native memory features and
object pointers.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor


class BoundedMemoryVideoPredictor(SAM2VideoPredictor):
    """Keep one prompt anchor plus a bounded set of native memory outputs.

    ``recent`` keeps the newest propagated outputs. ``confidence`` keeps the
    outputs with the highest native mask-decoder predicted IoU.  The latter
    uses MedSAM2's existing ``select_memory_by_iou`` read policy; it is not a
    learned reliability model.
    """

    def __init__(
        self,
        bounded_memory_policy: str = "recent",
        bounded_memory_size: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if bounded_memory_policy not in {"recent", "confidence"}:
            raise ValueError("bounded_memory_policy must be 'recent' or 'confidence'.")
        if bounded_memory_size < 1:
            raise ValueError("bounded_memory_size must be at least 1.")
        if bounded_memory_size > self.num_maskmem - 1:
            raise ValueError(
                "bounded_memory_size cannot exceed MedSAM2's available "
                f"non-conditioning slots ({self.num_maskmem - 1})."
            )

        self.bounded_memory_policy = bounded_memory_policy
        self.bounded_memory_size = bounded_memory_size
        # This is an existing MedSAM2 selection branch. It ranks the retained
        # native outputs by their decoder-predicted IoU before memory attention.
        self.select_memory_by_iou = bounded_memory_policy == "confidence"

    def propagate_in_video_preflight(self, inference_state):
        super().propagate_in_video_preflight(inference_state)
        condition_frames = inference_state["output_dict"]["cond_frame_outputs"]
        if len(condition_frames) != 1:
            raise ValueError(
                "Bounded-memory evaluation requires exactly one prompted anchor frame."
            )

    @torch.inference_mode()
    def propagate_in_video(self, *args, **kwargs) -> Iterator[tuple[int, list[int], torch.Tensor]]:
        """Propagate normally, then discard non-anchor entries after each frame.

        Pruning occurs when the caller advances the generator, so the caller can
        still save the current frame's output before a low-confidence current
        frame is discarded by the confidence policy.
        """
        inference_state = kwargs.get("inference_state", args[0] if args else None)
        if inference_state is None:
            raise TypeError("inference_state is required.")
        for result in super().propagate_in_video(*args, **kwargs):
            yield result
            self._prune_non_conditioning_memory(inference_state)

    def _prune_non_conditioning_memory(self, inference_state) -> None:
        outputs = inference_state["output_dict"]["non_cond_frame_outputs"]
        if len(outputs) <= self.bounded_memory_size:
            retained = sorted(outputs)
        elif self.bounded_memory_policy == "recent":
            retained = sorted(outputs)[-self.bounded_memory_size :]
        else:
            retained = [
                frame_idx
                for frame_idx, _ in sorted(
                    outputs.items(),
                    key=lambda item: (
                        -self._predicted_iou(item[1]),
                        -item[0],
                    ),
                )[: self.bounded_memory_size]
            ]
            retained.sort()

        retained_set = set(retained)
        for frame_idx in list(outputs):
            if frame_idx not in retained_set:
                outputs.pop(frame_idx)
        for object_outputs in inference_state["output_dict_per_obj"].values():
            per_object_non_cond = object_outputs["non_cond_frame_outputs"]
            for frame_idx in list(per_object_non_cond):
                if frame_idx not in retained_set:
                    per_object_non_cond.pop(frame_idx)

        inference_state["bounded_memory_trace"] = {
            "policy": self.bounded_memory_policy,
            "retained_non_cond_frame_indices": retained,
        }

    @staticmethod
    def _predicted_iou(output) -> float:
        predicted_iou = output.get("iou_predictions")
        if predicted_iou is None:
            return float("-inf")
        return float(predicted_iou.detach().float().mean().cpu())
