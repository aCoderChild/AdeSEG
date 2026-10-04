"""Uncertainty-aware Kalman spatial memory for frozen MedSAM2.

Replaces MedSAM2's memory bank with two constant-size slots:

- anchor slot: the prompt-frame memory, never overwritten;
- Kalman state: a mean memory map ``S`` and a per-pixel variance ``P``.

Per frame ``t`` (predict -> read -> observe -> update, as in RKN / KalmanNet)::

    predict:  P_prior = P_{t-1} + Q_t        Q_t = q * c(Q_net(F_t, |F_t - F_{t-1}|, log1p dt))
    read:     memory attention sees S_{t-1} + u * log1p(P_prior)
    observe:  C_t = MedSAM2 memory encoder(F_t, predicted mask_t)
    update:   R_t = (r + a * (1 - p_t)) * c(R_net(C_t, normalize(C_t - S_{t-1}), F_t, mask_t, p_t))
              K_t = P_prior / (P_prior + R_t)
              S_t = S_{t-1} + K_t * (C_t - S_{t-1})
              P_t = (1 - K_t) * P_prior

The transition is identity, the covariance is diagonal and shared across channels
at each pixel, and the observation model is identity (RKN's update with H = I).
The networks learn bounded corrections c(x) = range ** tanh(x) to the noise
levels q, r, a; they start at zero (c = 1), so the untrained module keeps
presence gating (an absent frame writes about 17% of a present one) and training
cannot remove it.
``KalmanMemoryMixin`` hooks MedSAM2's ``_prepare_memory_conditioned_features``
(predict + read) and ``_encode_memory_in_output`` (update), which both the video
predictor and ``SAM2Train`` call on every frame, so training and inference share
one implementation.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from modeling.native_pointers import append_native_object_pointers
from sam2.sam2_video_predictor import SAM2VideoPredictor


def _noise_head(in_channels: int, hidden: int) -> nn.Sequential:
    head = nn.Sequential(
        nn.Conv2d(in_channels, hidden, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(hidden, 1, 3, padding=1),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class KalmanMemoryUpdate(nn.Module):
    def __init__(
        self,
        memory_channels: int,
        image_channels: int,
        hidden_channels: int = 64,
        projection_channels: int = 32,
        initial_variance: float = 1.0,
        process_noise: float = 1.0,
        observation_noise: float = 0.1,
        absence_noise: float = 10.0,
        noise_range: float = 10.0,
        min_variance: float = 1e-4,
        max_variance: float = 1e4,
    ):
        super().__init__()
        self.config = {
            "hidden_channels": hidden_channels,
            "projection_channels": projection_channels,
            "initial_variance": initial_variance,
            "process_noise": process_noise,
            "observation_noise": observation_noise,
            "absence_noise": absence_noise,
            "noise_range": noise_range,
            "min_variance": min_variance,
            "max_variance": max_variance,
        }
        self.min_variance = min_variance
        self.max_variance = max_variance
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.absence_noise = absence_noise
        self.log_noise_range = math.log(noise_range)
        self.image_projection = nn.Conv2d(image_channels, projection_channels, 1)
        self.process_head = _noise_head(2 * projection_channels + 1, hidden_channels)
        self.observation_head = _noise_head(2 * memory_channels + projection_channels + 2, hidden_channels)
        self.log_initial_variance = nn.Parameter(torch.tensor(math.log(initial_variance)))
        self.uncertainty_embedding = nn.Parameter(torch.zeros(memory_channels))

    def correction(self, head_output: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_noise_range * torch.tanh(head_output))

    def project_image(self, image_feature: torch.Tensor) -> torch.Tensor:
        return self.image_projection(image_feature)

    def initial_variance(self, mean: torch.Tensor) -> torch.Tensor:
        return self.log_initial_variance.exp().expand(mean.size(0), 1, *mean.shape[-2:])

    def readout(self, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        return mean + self.uncertainty_embedding.view(1, -1, 1, 1) * torch.log1p(variance)

    def predict(self, variance, image_projection, previous_image_projection, frame_gap):
        batch, _, height, width = variance.shape
        log_gap = torch.log1p(frame_gap.reshape(batch, 1, 1, 1).clamp_min(0.0))
        process_noise = self.process_noise * self.correction(self.process_head(torch.cat([
            image_projection,
            (image_projection - previous_image_projection).abs(),
            log_gap.expand(batch, 1, height, width),
        ], dim=1)))
        prior_variance = (variance + process_noise).clamp(self.min_variance, self.max_variance)
        return prior_variance, process_noise

    def update(self, mean, prior_variance, candidate, image_projection, mask_probability, presence_probability):
        if mean.shape != candidate.shape or mean.ndim != 4:
            raise ValueError("mean and candidate must be matching [B, C, H, W] tensors.")
        batch, _, height, width = mean.shape
        mask_probability = F.interpolate(mask_probability, size=(height, width), mode="area")
        presence = presence_probability.reshape(batch, 1, 1, 1)
        innovation = candidate - mean
        observation_noise = (self.observation_noise + self.absence_noise * (1.0 - presence)) * self.correction(
            self.observation_head(torch.cat([
                candidate,
                F.normalize(innovation, dim=1),
                image_projection,
                mask_probability,
                presence.expand(batch, 1, height, width),
            ], dim=1))
        )
        gain = prior_variance / (prior_variance + observation_noise)
        return {
            "mean": mean + gain * innovation,
            "variance": ((1.0 - gain) * prior_variance).clamp(self.min_variance, self.max_variance),
            "gain": gain,
            "observation_noise": observation_noise,
        }


class KalmanState:
    """Anchor slot plus Kalman mean/variance."""

    def __init__(self, anchor, position, image_projection, frame_idx, frame_time, update: KalmanMemoryUpdate):
        self.anchor = anchor.float()
        self.position = position.float()
        self.mean = self.anchor
        self.variance = update.initial_variance(self.mean)
        self.image_projection = image_projection
        self.last_frame_idx = frame_idx
        self.last_time = frame_time
        self.updates = 0
        self.detection = None

    def tensors(self) -> list[torch.Tensor]:
        tensors = [self.anchor, self.position, self.mean, self.variance, self.image_projection]
        return tensors if self.detection is None else tensors + [self.detection]


class KalmanMemoryMixin:
    """Put before a ``SAM2Base`` subclass; inactive while ``memory_update`` is None or ``kalman_enabled`` is False."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_update = None
        self.kalman_enabled = True
        self.kalman_frame_times = None
        self.kalman_last_step = None
        self._kalman_step = None
        self.kalman_observation_frame = False

    def _kalman_active(self) -> bool:
        return self.kalman_enabled and self.memory_update is not None

    def _kalman_time(self, frame_idx: int) -> float:
        times = self.kalman_frame_times
        return float(frame_idx if times is None else times[frame_idx])

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        step = {"output_dict": output_dict, "frame_idx": frame_idx, "init": is_init_cond_frame}
        if self._kalman_active():
            batch = current_vision_feats[-1].size(1)
            image_feature = current_vision_feats[-1].permute(1, 2, 0).reshape(batch, -1, *feat_sizes[-1])
            step["projection"] = self.memory_update.project_image(image_feature.float())
        if not self._kalman_active() or is_init_cond_frame:
            step["pix_feat"] = super()._prepare_memory_conditioned_features(
                frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
                feat_sizes, output_dict, num_frames, track_in_reverse,
            )
            self._kalman_step = step
            return step["pix_feat"]
        if track_in_reverse:
            raise ValueError("Kalman memory supports forward propagation only.")
        state = output_dict["kalman_state"]
        if frame_idx != state.last_frame_idx + 1:
            raise ValueError("Kalman memory requires consecutive forward frames.")
        frame_time = self._kalman_time(frame_idx)
        gap = torch.full((batch,), frame_time - state.last_time, device=state.mean.device)
        step["prior_variance"], step["process_noise"] = self.memory_update.predict(
            state.variance, step["projection"], state.image_projection, gap
        )
        step["observation"] = self.kalman_observation_frame
        if step["observation"]:
            # Detector-prompted frame: decode on memory-free features, as on a prompt frame.
            step["pix_feat"] = super()._prepare_memory_conditioned_features(
                frame_idx, True, current_vision_feats, current_vision_pos_embeds,
                feat_sizes, output_dict, num_frames, track_in_reverse,
            )
            self._kalman_step = step
            return step["pix_feat"]
        dtype = current_vision_feats[-1].dtype
        memory_chunks, position_chunks = kalman_memory_tokens(
            self, state.anchor, self.memory_update.readout(state.mean, step["prior_variance"]),
            state.position, dtype, state.detection,
        )
        num_obj_ptr_tokens = append_native_object_pointers(
            self, frame_idx, output_dict, num_frames, track_in_reverse,
            current_vision_feats[-1].device, batch, memory_chunks, position_chunks,
        )
        fused = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=torch.cat(memory_chunks, dim=0),
            memory_pos=torch.cat(position_chunks, dim=0),
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        step["pix_feat"] = fused.permute(1, 2, 0).reshape(batch, self.hidden_dim, *feat_sizes[-1])
        self._kalman_step = step
        return step["pix_feat"]

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        super()._encode_memory_in_output(
            current_vision_feats, feat_sizes, point_inputs, run_mem_encoder,
            high_res_masks, object_score_logits, current_out,
        )
        if current_out["maskmem_features"] is not None:
            # MedSAM2's video predictor stores frame memories in bf16; round here so training matches it.
            features = current_out["maskmem_features"]
            current_out["maskmem_features"] = features.to(torch.bfloat16).to(features.dtype)
        step, self._kalman_step = self._kalman_step, None
        if step is None:
            return
        current_out["pix_feat_with_mem"] = step["pix_feat"]
        candidate = current_out.get("maskmem_features")
        if not self._kalman_active() or candidate is None:
            return
        output_dict, frame_idx = step["output_dict"], step["frame_idx"]
        if step["init"]:
            output_dict["kalman_state"] = KalmanState(
                candidate, current_out["maskmem_pos_enc"][-1], step["projection"],
                frame_idx, self._kalman_time(frame_idx), self.memory_update,
            )
            return
        state = output_dict["kalman_state"]
        presence = torch.sigmoid(object_score_logits.float())
        updated = self.memory_update.update(
            state.mean,
            step["prior_variance"],
            candidate.float(),
            step["projection"],
            torch.sigmoid(current_out["pred_masks"].float()),
            presence,
        )
        if not (torch.isfinite(updated["mean"]).all() and torch.isfinite(updated["variance"]).all()):
            raise FloatingPointError(f"Kalman state became non-finite at frame {frame_idx}.")
        state.mean, state.variance = updated["mean"], updated["variance"]
        if step.get("observation") and bool((presence > 0.5).all()):
            state.detection = candidate.float()
        state.image_projection = step["projection"]
        state.last_frame_idx = frame_idx
        state.last_time = self._kalman_time(frame_idx)
        state.updates += 1
        updated.update(
            presence=presence,
            prior_variance=step["prior_variance"],
            process_noise=step["process_noise"],
        )
        current_out["kalman"] = updated
        self.kalman_last_step = updated


def _wait_for_offload(device: torch.device) -> None:
    # MedSAM2 offloads with non_blocking=True; on MPS the host copy can still be in flight.
    if device.type == "mps":
        torch.mps.synchronize()


class KalmanMemoryVideoPredictor(KalmanMemoryMixin, SAM2VideoPredictor):
    """MedSAM2 video predictor with the anchor slot plus Kalman state."""

    def propagate_in_video_preflight(self, inference_state):
        if not inference_state.get("kalman_enabled", False):
            raise ValueError("Kalman memory must be enabled before propagation.")
        if self._get_obj_num(inference_state) != 1:
            raise ValueError("Kalman memory currently supports exactly one object.")
        if self.memory_update is None:
            raise RuntimeError("Kalman memory requires a loaded checkpoint.")
        super().propagate_in_video_preflight(inference_state)
        outputs = inference_state["output_dict"]
        anchor_idx = inference_state["kalman_anchor_frame_idx"]
        if set(outputs["cond_frame_outputs"]) != {anchor_idx}:
            raise ValueError("Kalman memory supports one initial prompt frame.")
        self.kalman_frame_times = inference_state.get("frame_times")
        anchor = outputs["cond_frame_outputs"][anchor_idx]
        device = anchor["obj_ptr"].device
        _wait_for_offload(device)
        _, backbone_out, _, _, _ = self._get_image_feature(inference_state, anchor_idx, 1)
        outputs["kalman_state"] = KalmanState(
            anchor["maskmem_features"].to(device),
            anchor["maskmem_pos_enc"][-1].to(device),
            self.memory_update.project_image(backbone_out["backbone_fpn"][-1].float()),
            anchor_idx,
            self._kalman_time(anchor_idx),
            self.memory_update,
        )
        anchor["maskmem_features"] = None
        anchor["maskmem_pos_enc"] = None

    def _reset_tracking_results(self, inference_state):
        super()._reset_tracking_results(inference_state)
        inference_state["output_dict"].pop("kalman_state", None)

    def _run_single_frame_inference(self, *args, **kwargs):
        self.kalman_last_step = None
        current_out, pred_masks = super()._run_single_frame_inference(*args, **kwargs)
        step = self.kalman_last_step
        if step is not None and not kwargs["is_init_cond_frame"]:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None
            current_out["kalman_trace"] = {
                "presence": float(step["presence"].mean()),
                "gain_mean": float(step["gain"].mean()),
                "prior_variance_mean": float(step["prior_variance"].mean()),
                "variance_mean": float(step["variance"].mean()),
                "observation_noise_mean": float(step["observation_noise"].mean()),
                "process_noise_mean": float(step["process_noise"].mean()),
            }
        return current_out, pred_masks


def kalman_memory_tokens(model, anchor, state_readout, position, dtype=None, detection=None):
    """Anchor (and detection slot) with the conditioning-frame temporal code, state with the latest-frame code."""
    dtype = dtype or anchor.dtype
    anchor_position = position + model.maskmem_tpos_enc[model.num_maskmem - 1].view(1, -1, 1, 1)
    state_position = position + model.maskmem_tpos_enc[0].view(1, -1, 1, 1)
    tokens = lambda tensor: tensor.to(dtype).flatten(2).permute(2, 0, 1)
    memory, positions = [tokens(anchor), tokens(state_readout)], [tokens(anchor_position), tokens(state_position)]
    if detection is not None:
        memory.insert(1, tokens(detection))
        positions.insert(1, tokens(anchor_position))
    return memory, positions
