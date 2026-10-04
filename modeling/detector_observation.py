"""Detector boxes as per-frame observations for MedSAM2 video propagation.

On a frame with a confident detection, the box prompts the decoder and the
resulting mask is written to memory as that frame's observation.

- ``observation_clean``: decode the box on memory-free features, as on a prompt
  frame, so a drifted memory cannot pull the observation off the polyp. The
  Kalman memory also keeps the latest such observation in a detection slot.
- ``observation_presence``: the detector votes on presence; its log-odds are
  added to the decoder's object score (two independent sensors, equal prior).
"""

from __future__ import annotations

import math

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import concat_points

from modeling.kalman_memory import KalmanMemoryVideoPredictor


class DetectorObservationMixin:
    """``observations``: {frame_idx: (xyxy box in video pixels, confidence)}, set per video."""

    observations = None
    observation_presence = True
    observation_clean = True

    def _run_single_frame_inference(self, *args, **kwargs):
        observation = (self.observations or {}).get(kwargs["frame_idx"])
        if observation is None or kwargs["point_inputs"] is not None or kwargs["is_init_cond_frame"]:
            return super()._run_single_frame_inference(*args, **kwargs)
        box, confidence = observation
        prompt = self.prepare_point_inputs(inference_state=kwargs["inference_state"], box=box)
        kwargs["point_inputs"] = concat_points(None, prompt["point_coords"], prompt["point_labels"])
        if self.observation_presence:
            confidence = min(max(confidence, 1e-4), 1 - 1e-4)
            self.object_score_bias = math.log(confidence / (1 - confidence))
        if self.observation_clean:
            self._mark_clean_observation(kwargs)
        try:
            return super()._run_single_frame_inference(*args, **kwargs)
        finally:
            self.object_score_bias = None
            self.kalman_observation_frame = False

    def _mark_clean_observation(self, kwargs):
        kwargs["is_init_cond_frame"] = True


class ObservedVideoPredictor(DetectorObservationMixin, SAM2VideoPredictor):
    pass


class ObservedKalmanVideoPredictor(DetectorObservationMixin, KalmanMemoryVideoPredictor):
    def _mark_clean_observation(self, kwargs):
        self.kalman_observation_frame = True
