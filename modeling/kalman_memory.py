"""Uncertainty-aware Kalman spatial memory for frozen MedSAM2.

Replaces MedSAM2's memory bank with two constant-size slots:

- anchor slot: the prompt-frame memory, never overwritten;
- Kalman state: a mean memory map ``S`` and a per-pixel variance ``P``.

Per frame ``t`` (predict -> read -> observe -> update, as in RKN / KalmanNet)::

    predict:  P_prior = P_{t-1} + q
    read:     memory attention sees S_{t-1} + u * log1p(P_prior)
    observe:  C_t = MedSAM2 memory encoder(F_t, predicted mask_t)
    update:   R_t = (r + a * (1 - p_t)) * c(D_t - mean(D_t)) * [c(g(logit d_t)) on detector frames]
              D_t = R_net(C_t, normalize(C_t - S_{t-1}), F_t, mask_t, p_t)
              K_t = P_prior / (P_prior + R_t)
              S_t = S_{t-1} + K_t * (C_t - S_{t-1})
              P_t = (1 - K_t) * P_prior

The transition is identity, the covariance is diagonal and shared across channels
at each pixel, and the observation model is identity (RKN's update with H = I).
The overall write level is hand-set: q, r and the absence noise a (an absent
frame writes about 17% of a present one). Learning only redistributes it, with
bounded corrections c(x) = range ** tanh(x):

- R_net's map is centred per frame, so it decides where in the frame the new
  observation is trusted (polyp vs. highlights, occluders, blur), not how much
  the frame is written overall;
- g maps the detector confidence d_t to how much a detector observation is trusted.

Both start at zero (c = 1), so the untrained module is the hand-set filter.
Pseudo-videos made from single frames cannot teach how fast appearance changes,
which is why the overall write level is not learned.
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

CHECKPOINT_FORMAT = "adseg_kalman_memory_v4"


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
        self.memory_channels = memory_channels
        self.image_channels = image_channels
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
        self.observation_head = _noise_head(2 * memory_channels + projection_channels + 2, hidden_channels)
        self.detection_trust = nn.Linear(1, 1)
        nn.init.zeros_(self.detection_trust.weight)
        nn.init.zeros_(self.detection_trust.bias)
        self.register_buffer("log_initial_variance", torch.tensor(math.log(initial_variance)))
        self.uncertainty_embedding = nn.Parameter(torch.zeros(memory_channels))

    def correction(self, head_output: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_noise_range * torch.tanh(head_output))

    def project_image(self, image_feature: torch.Tensor) -> torch.Tensor:
        return self.image_projection(image_feature)

    def initial_variance(self, mean: torch.Tensor) -> torch.Tensor:
        return self.log_initial_variance.exp().expand(mean.size(0), 1, *mean.shape[-2:])

    def readout(self, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        return mean + self.uncertainty_embedding.view(1, -1, 1, 1) * torch.log1p(variance)

    def predict(self, variance, noise_scale=None):
        """``noise_scale``: optional per-pixel factor on q (e.g. measured feature change / its dev mean)."""
        process_noise = self.process_noise if noise_scale is None else self.process_noise * noise_scale
        return (variance + process_noise).clamp(self.min_variance, self.max_variance)

    def update(
        self, mean, prior_variance, candidate, image_projection, mask_probability, presence_probability,
        detection_logit=None, reliability=None, noise_scale=None,
    ):
        """``reliability``: calibrated probability [B] that the frame is correctly tracked. When given, the
        observation noise is q * (1 - rho) / rho, so rho = 0.5 trusts the frame as much as the memory.
        ``noise_scale``: optional per-pixel factor on the observation noise (e.g. camera motion)."""
        if mean.shape != candidate.shape or mean.ndim != 4:
            raise ValueError("mean and candidate must be matching [B, C, H, W] tensors.")
        batch, _, height, width = mean.shape
        mask_probability = F.interpolate(mask_probability, size=(height, width), mode="area")
        presence = presence_probability.reshape(batch, 1, 1, 1)
        innovation = candidate - mean
        spatial = self.observation_head(torch.cat([
            candidate,
            F.normalize(innovation, dim=1),
            image_projection,
            mask_probability,
            presence.expand(batch, 1, height, width),
        ], dim=1))
        base = self.observation_noise + self.absence_noise * (1.0 - presence)
        if reliability is not None:
            rho = reliability.reshape(batch, 1, 1, 1).clamp(1e-3, 1.0 - 1e-3)
            base = self.process_noise * (1.0 - rho) / rho
        observation_noise = base * self.correction(spatial - spatial.mean(dim=(2, 3), keepdim=True))
        if noise_scale is not None:
            observation_noise = observation_noise * noise_scale
        if detection_logit is not None:
            trust = self.detection_trust(torch.full((batch, 1), detection_logit, device=mean.device))
            observation_noise = observation_noise * self.correction(trust).view(batch, 1, 1, 1)
        gain = prior_variance / (prior_variance + observation_noise)
        return {
            "mean": mean + gain * innovation,
            "variance": ((1.0 - gain) * prior_variance).clamp(self.min_variance, self.max_variance),
            "gain": gain,
            "observation_noise": observation_noise,
        }


def mask_prototype(features: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
    """Mask-weighted mean of a memory map [B, C, H, W]: the object's appearance, independent of position."""
    weights = F.interpolate(torch.sigmoid(mask_logits.float()), size=features.shape[-2:], mode="area")
    return (features.float() * weights).sum(dim=(2, 3)) / weights.sum(dim=(2, 3)).clamp_min(1e-6)


def anchor_signals(memory, mask_logits, pointer, image_feature) -> dict[str, torch.Tensor]:
    """Object descriptors compared with the prompt frame's (diagnostics: memory, pointer and image appearance)."""
    return {
        "memory": mask_prototype(memory, mask_logits),
        "pointer": pointer.float(),
        "image": mask_prototype(image_feature, mask_logits),
    }


def vision_feature_map(current_vision_feats, feat_sizes) -> torch.Tensor:
    feature = current_vision_feats[-1]
    return feature.permute(1, 2, 0).reshape(feature.size(1), -1, *feat_sizes[-1])


class KalmanState:
    """Anchor slot plus Kalman mean/variance, and the prototype of the last accepted object memory."""

    def __init__(self, anchor, position, frame_idx, update: KalmanMemoryUpdate, prototype):
        self.anchor = anchor.float()
        self.position = position.float()
        self.mean = self.anchor
        self.variance = update.initial_variance(self.mean)
        self.last_frame_idx = frame_idx
        self.updates = 0
        self.detection = None
        self.prototype = prototype
        self.anchor_signals = None  # prompt-frame descriptors for the anchor distances (diagnostics)
        self.previous_feature = None  # image features of the previous frame, for the camera-motion measure

    def tensors(self) -> list[torch.Tensor]:
        tensors = [self.anchor, self.position, self.mean, self.variance]
        return tensors if self.detection is None else tensors + [self.detection]


class KalmanMemoryMixin:
    """Put before a ``SAM2Base`` subclass; inactive while ``memory_update`` is None or ``kalman_enabled`` is False."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_update = None
        self.kalman_enabled = True
        self.kalman_last_step = None
        self._kalman_step = None
        self.kalman_observation_frame = False
        self.kalman_observation_logit = None
        # Off by default (the reported DOK-Mem). skip_absent: a frame whose presence gate is closed is a
        # missing measurement (predict only). gate: a tracker frame whose object prototype is farther than
        # this cosine distance from the last accepted one is rejected; confident detections are never gated.
        self.kalman_skip_absent = False
        self.kalman_gate = None
        # Optional (a, c): reliability rho = sigmoid(a * object score + c), calibrated on dev videos
        # (scripts/calibrate_reliability.py); replaces the presence-only observation noise.
        self.kalman_reliability = None
        # Optional d_ref: q scales per pixel with the change of MedSAM2's image features since the previous frame
        # (1 - cosine per pixel) divided by d_ref, its mean on dev videos; camera jumps then raise P where the view changed.
        self.kalman_motion_noise = None
        # Ablation (g_present, g_absent): replace the Kalman gain by constant gains, chosen by presence > 0.5 only,
        # i.e. a presence-gated exponential moving average of the frame memories.
        self.kalman_fixed_gain = None
        # Optional d_ref: camera motion as measurement noise. R scales per pixel with the image-feature change
        # since the previous frame divided by d_ref (its dev mean), so frames taken during a jump write less.
        self.kalman_motion_measurement = None
        # Diagnostic upper bound, never a method: {frame_idx: binary ground-truth mask}. A frame is written only
        # if it is correct (IoU >= 0.5 with the ground truth, or correctly empty); otherwise K = 0.
        self.kalman_oracle_masks = None
        # Optional tau: measurement validation. A frame whose object-score logit is below tau is not written
        # (K = 0, the state is kept and its variance grows), as an outlier-rejecting Kalman filter does.
        self.kalman_score_gate = None
        self.kalman_score_gate_present_only = False  # gate only frames called present (0 < score < tau)

    def _kalman_active(self) -> bool:
        return self.kalman_enabled and self.memory_update is not None

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
        feature = vision_feature_map(current_vision_feats, feat_sizes).float()
        change = None
        if state.previous_feature is not None:
            change = 1.0 - F.cosine_similarity(feature, state.previous_feature, dim=1).unsqueeze(1)
        state.previous_feature = feature
        step["feature_change"] = change
        scale = None if change is None or self.kalman_motion_noise is None else change / self.kalman_motion_noise
        step["prior_variance"] = self.memory_update.predict(state.variance, scale)
        step["observation"] = self.kalman_observation_frame
        step["observation_logit"] = self.kalman_observation_logit if step["observation"] else None
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
                candidate, current_out["maskmem_pos_enc"][-1], frame_idx, self.memory_update,
                mask_prototype(candidate, current_out["pred_masks"]),
            )
            output_dict["kalman_state"].previous_feature = vision_feature_map(current_vision_feats, feat_sizes).float()
            return
        state = output_dict["kalman_state"]
        presence = torch.sigmoid(object_score_logits.float())
        prototype = mask_prototype(candidate, current_out["pred_masks"])
        distance = 1.0 - F.cosine_similarity(prototype, state.prototype, dim=1)
        status = "accepted"
        if self.kalman_skip_absent and bool((presence <= 0.5).all()):
            status = "absent"
        elif self.kalman_gate is not None and not step.get("observation") and bool((distance > self.kalman_gate).all()):
            status = "rejected"
        # The gate is calibrated on MedSAM2's raw object score; a presence fusion may have replaced it.
        raw = getattr(self, "raw_object_score", None)
        score = raw if raw is not None else float(object_score_logits.float().mean())
        if self.kalman_score_gate is not None and score < self.kalman_score_gate:
            if not self.kalman_score_gate_present_only or score > 0:
                status = "rejected"
        if self.kalman_oracle_masks is not None and frame_idx in self.kalman_oracle_masks:
            truth = self.kalman_oracle_masks[frame_idx]
            predicted = current_out["pred_masks"][0, 0] > 0
            truth = F.interpolate(truth[None, None].float(), size=predicted.shape, mode="nearest")[0, 0] > 0
            union = (predicted | truth).sum()
            correct = (not truth.any() and not predicted.any()) or (union > 0 and (predicted & truth).sum() / union >= 0.5)
            status = "accepted" if correct else "rejected"
        reliability = None
        if self.kalman_reliability is not None:
            slope, offset = self.kalman_reliability
            reliability = torch.sigmoid(slope * object_score_logits.float() + offset)
        updated = self.memory_update.update(
            state.mean,
            step["prior_variance"],
            candidate.float(),
            step["projection"],
            torch.sigmoid(current_out["pred_masks"].float()),
            presence,
            step["observation_logit"],
            reliability,
            None if self.kalman_motion_measurement is None or step.get("feature_change") is None
            else step["feature_change"] / self.kalman_motion_measurement,
        )
        if self.kalman_fixed_gain is not None:
            constant = self.kalman_fixed_gain[0] if bool((presence > 0.5).all()) else self.kalman_fixed_gain[1]
            gain = torch.full_like(updated["gain"], constant)
            updated.update(mean=state.mean + gain * (candidate.float() - state.mean), gain=gain)
        if status != "accepted":  # missing measurement: keep the mean, let the variance grow
            updated.update(mean=state.mean, variance=step["prior_variance"], gain=torch.zeros_like(updated["gain"]))
        if not (torch.isfinite(updated["mean"]).all() and torch.isfinite(updated["variance"]).all()):
            raise FloatingPointError(f"Kalman state became non-finite at frame {frame_idx}.")
        state.mean, state.variance = updated["mean"], updated["variance"]
        if status == "accepted" and bool((presence > 0.5).all()):
            state.prototype = prototype
            if step.get("observation"):
                state.detection = candidate.float()
        state.last_frame_idx = frame_idx
        state.updates += 1
        updated.update(presence=presence, prior_variance=step["prior_variance"], status=status, distance=distance)
        if step.get("feature_change") is not None:
            updated["feature_change"] = step["feature_change"]
        if state.anchor_signals is not None:
            current = anchor_signals(candidate, current_out["pred_masks"], current_out["obj_ptr"],
                                     vision_feature_map(current_vision_feats, feat_sizes))
            updated["anchor_distance"] = {
                name: 1.0 - F.cosine_similarity(current[name], state.anchor_signals[name], dim=1) for name in current
            }
        current_out["kalman"] = updated
        current_out["kalman_rejected"] = status == "rejected"
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
        anchor = outputs["cond_frame_outputs"][anchor_idx]
        device = anchor["obj_ptr"].device
        _wait_for_offload(device)
        features = anchor["maskmem_features"].to(device)
        outputs["kalman_state"] = KalmanState(
            features, anchor["maskmem_pos_enc"][-1].to(device), anchor_idx, self.memory_update,
            mask_prototype(features, anchor["pred_masks"].to(device)),
        )
        _, _, vision_feats, _, feat_sizes = self._get_image_feature(inference_state, anchor_idx, 1)
        anchor_feature = vision_feature_map(vision_feats, feat_sizes)
        outputs["kalman_state"].anchor_signals = anchor_signals(
            features, anchor["pred_masks"].to(device), anchor["obj_ptr"].to(device), anchor_feature,
        )
        outputs["kalman_state"].previous_feature = anchor_feature.float()
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
            current_out["kalman_rejected"] = step["status"] == "rejected"
            current_out["kalman_trace"] = {
                "kalman_status": step["status"],
                "innovation_distance": float(step["distance"].mean()),
                **{f"anchor_distance_{name}": float(value.mean()) for name, value in step.get("anchor_distance", {}).items()},
                "presence": float(step["presence"].mean()),
                "gain_mean": float(step["gain"].mean()),
                "prior_variance_mean": float(step["prior_variance"].mean()),
                "feature_change_mean": float(step["feature_change"].mean()) if "feature_change" in step else float("nan"),
                "variance_mean": float(step["variance"].mean()),
                "observation_noise_mean": float(step["observation_noise"].mean()),
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


def save_memory_update(update: KalmanMemoryUpdate, path) -> None:
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "memory_channels": update.memory_channels,
        "image_channels": update.image_channels,
        "update_config": update.config,
        "state_dict": update.state_dict(),
    }, path)


def load_memory_update(path, model, device) -> KalmanMemoryUpdate:
    """Kalman update from ``save_memory_update``, checked against ``model``'s MedSAM2 widths."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint.")
    if (checkpoint["memory_channels"], checkpoint["image_channels"]) != (model.mem_dim, model.hidden_dim):
        raise ValueError(f"{path} does not match this MedSAM2 model.")
    update = KalmanMemoryUpdate(model.mem_dim, model.hidden_dim, **checkpoint["update_config"]).to(device)
    update.load_state_dict(checkpoint["state_dict"])
    return update.eval()
