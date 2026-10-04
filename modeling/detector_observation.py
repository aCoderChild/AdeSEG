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
- ``presence_filter``: the per-frame presence log-odds become measurements of a
  presence belief filtered over time (``PresenceFilter``); the gate uses the belief.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

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


class PresenceFilter(nn.Module):
    """Scalar Kalman filter on presence log-odds, the presence part of the memory state.

    predict:  b- = a * b,  v- = a^2 * v + softplus(q0 + q_gap * log1p(gap) + q_var * log P_prior)
    update:   K = v- / (v- + r),  b = b- + K * (clip(z) - b-),  v = (1 - K) * v-
    with z the per-frame presence log-odds (fusion head, or detector log-odds on
    frames with a confident box) and r learned separately for the two cases.
    """

    def __init__(self, measurement_limit: float = 10.0):
        super().__init__()
        self.measurement_limit = measurement_limit
        self.raw_decay = nn.Parameter(torch.tensor(3.0))
        self.process = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.raw_noise = nn.Parameter(torch.tensor([0.0, 0.0]))  # unobserved, observed
        self.initial = nn.Parameter(torch.tensor([5.0, 0.0]))  # prompt-frame log-odds, raw variance

    def start(self, batch: int = 1):
        b = self.initial[0].expand(batch)
        return b, F.softplus(self.initial[1]).expand(batch)

    def forward(self, b, v, z, observed, gap, log_prior_variance):
        decay = torch.sigmoid(self.raw_decay)
        q = F.softplus(self.process[0] + self.process[1] * torch.log1p(gap) + self.process[2] * log_prior_variance)
        b_prior, v_prior = decay * b, decay**2 * v + q
        r = F.softplus(torch.where(observed, self.raw_noise[1], self.raw_noise[0]))
        gain = v_prior / (v_prior + r)
        z = z.clamp(-self.measurement_limit, self.measurement_limit)
        return b_prior + gain * (z - b_prior), (1 - gain) * v_prior


class DetectorObservationMixin:
    """``observations``: {frame_idx: (xyxy box in video pixels, confidence)}, set per video."""

    observations = None
    observation_conf = 0.5
    observation_presence = True
    observation_clean = True
    record_ungated_mask = False  # keep the decoder's mask before the presence gate (fusion training data)

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

        def hook(object_score, ious, masks):
            step = getattr(self, "_kalman_step", None) or {}
            variance = step.get("prior_variance")
            features = torch.stack([
                object_score[:, 0].float(),
                ious.max(dim=1).values.float(),
                torch.full_like(object_score[:, 0].float(), logit),
                torch.full_like(object_score[:, 0].float(), 0.0 if variance is None else float(variance.mean().log())),
            ], dim=1)
            trace.update((f"feature_{name}", value) for name, value in zip(PRESENCE_FEATURES, features[0].tolist()))
            if self.record_ungated_mask:
                self.ungated_mask = masks[torch.arange(len(masks)), ious.argmax(dim=1)].float()
            if observed:
                fused = object_score + logit if self.observation_presence else object_score
            elif fusion is not None:
                fused = fusion(features).to(object_score.dtype)
            else:
                fused = object_score
            presence_filter = getattr(self, "presence_filter", None)
            if presence_filter is not None:
                state = kwargs["inference_state"]
                b, v = state.get("presence_belief") or presence_filter.start(len(fused))
                times, index = state.get("frame_times"), kwargs["frame_idx"]
                gap = float(times[index] - times[index - 1]) if times else 1.0
                z = fused[:, 0].float()
                b, v = presence_filter(
                    b.to(z.device), v.to(z.device), z, torch.full_like(z, float(observed)) > 0,
                    torch.full_like(z, gap), features[:, 3],
                )
                state["presence_belief"] = (b, v)
                trace["presence_measurement"] = float(fused.float().mean())
                fused = b[:, None].to(object_score.dtype)
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
