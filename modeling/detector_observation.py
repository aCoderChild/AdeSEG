"""Detector boxes as per-frame observations for MedSAM2 video propagation.

``observations`` holds the top detection of every frame. On a frame whose top
detection reaches ``observation_conf``, the box prompts the decoder and the
resulting mask is written to memory as that frame's observation.

- ``observation_clean``: decode the box on memory-free features, as on a prompt
  frame, so a drifted memory cannot pull the observation off the polyp. The
  Kalman memory also keeps the latest such observation in a detection slot.
- ``observation_presence``: on observed frames the detector's log-odds are added
  to the decoder's object score (two independent sensors, equal prior).
- ``presence_fusion``: on the other frames a learned head maps the decoder's
  object score and IoU, the (weak) detector evidence and the Kalman prior
  variance to the object score that gates the mask and the memory. Frames the
  detector did not run on (not in ``observations``) use
  ``presence_fusion_unmeasured``, a head fitted without the detector feature.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import concat_points

from modeling.kalman_memory import KalmanMemoryVideoPredictor

PRESENCE_FEATURES = ("object_score", "predicted_iou", "detector_logit", "log_prior_variance")


def detector_logit(confidence: float) -> float:
    confidence = min(max(confidence, 1e-4), 1 - 1e-4)
    return math.log(confidence / (1 - confidence))


class PresenceFusion(nn.Module):
    """Logistic presence fusion; initialized to pass the decoder's object score through."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(len(PRESENCE_FEATURES), 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        with torch.no_grad():
            self.linear.weight[0, 0] = 1.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class DetectorObservationMixin:
    """``observations``: {frame_idx: (xyxy box in video pixels, confidence)}, set per video."""

    observations = None
    observation_conf = 0.5
    observation_presence = True
    observation_clean = True

    def _run_single_frame_inference(self, *args, **kwargs):
        if kwargs["point_inputs"] is not None or kwargs["is_init_cond_frame"]:
            return super()._run_single_frame_inference(*args, **kwargs)
        measured = kwargs["frame_idx"] in (self.observations or {})
        box, confidence = self.observations[kwargs["frame_idx"]] if measured else (None, 0.0)
        observed = box is not None and confidence >= self.observation_conf
        if observed:
            prompt = self.prepare_point_inputs(inference_state=kwargs["inference_state"], box=box)
            kwargs["point_inputs"] = concat_points(None, prompt["point_coords"], prompt["point_labels"])
            if self.observation_clean:
                self._mark_clean_observation(kwargs)
        trace = {"observed": int(observed)}
        logit = detector_logit(confidence) if measured else 0.0
        fusion = getattr(self, "presence_fusion" if measured else "presence_fusion_unmeasured", None)

        def hook(object_score, ious):
            step = getattr(self, "_kalman_step", None) or {}
            variance = step.get("prior_variance")
            features = torch.stack([
                object_score[:, 0].float(),
                ious.max(dim=1).values.float(),
                torch.full_like(object_score[:, 0].float(), logit),
                torch.full_like(object_score[:, 0].float(), 0.0 if variance is None else float(variance.mean().log())),
            ], dim=1)
            trace.update((f"feature_{name}", value) for name, value in zip(PRESENCE_FEATURES, features[0].tolist()))
            if observed:
                fused = object_score + logit if self.observation_presence else object_score
            elif fusion is not None:
                fused = fusion(features).to(object_score.dtype)
            else:
                fused = object_score
            trace["fused_score"] = float(fused.float().mean())
            return fused

        self.object_score_hook = hook
        try:
            current_out, pred_masks = super()._run_single_frame_inference(*args, **kwargs)
        finally:
            self.object_score_hook = None
            self.kalman_observation_frame = False
        current_out["presence_trace"] = trace
        return current_out, pred_masks

    def _mark_clean_observation(self, kwargs):
        kwargs["is_init_cond_frame"] = True


class ObservedVideoPredictor(DetectorObservationMixin, SAM2VideoPredictor):
    pass


class ObservedKalmanVideoPredictor(DetectorObservationMixin, KalmanMemoryVideoPredictor):
    def _mark_clean_observation(self, kwargs):
        self.kalman_observation_frame = True
