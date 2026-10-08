"""Uncertainty-aware Kalman spatial memory for frozen MedSAM2."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from modeling.native_pointers import append_native_object_pointers
from sam2.sam2_video_predictor import SAM2VideoPredictor

CHECKPOINT_FORMAT = "adseg_kalman_memory_v5"
LEGACY_CHECKPOINT_FORMAT = "adseg_kalman_memory_v4"


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

    def predict(self, variance):
        """Kalman predict: P⁻ = P + q (identity transition)."""
        return (variance + self.process_noise).clamp(self.min_variance, self.max_variance)

    def update(self, mean, prior_variance, candidate, image_projection, mask_probability, presence_probability):
        """Kalman update with presence-dependent measurement noise.

        R_t = (r + a·(1 − p_t)) · c(spatial map), so an absent frame (p_t → 0) is
        trusted far less than a present one; c = 10^tanh starts at 1 (untrained).
        K = P⁻/(P⁻ + R); S ← S + K(C_t − S); P ← (1 − K)P⁻.
        """
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
        observation_noise = base * self.correction(spatial - spatial.mean(dim=(2, 3), keepdim=True))
        gain = prior_variance / (prior_variance + observation_noise)
        return {
            "mean": mean + gain * innovation,
            "variance": ((1.0 - gain) * prior_variance).clamp(self.min_variance, self.max_variance),
            "gain": gain,
            "observation_noise": observation_noise,
        }


def vision_feature_map(current_vision_feats, feat_sizes) -> torch.Tensor:
    feature = current_vision_feats[-1]
    return feature.permute(1, 2, 0).reshape(feature.size(1), -1, *feat_sizes[-1])


class KalmanState:
    """Fixed prompt anchor slot plus the Kalman mean/variance of the spatial memory."""

    def __init__(self, anchor, position, frame_idx, update: KalmanMemoryUpdate):
        self.anchor = anchor.float()
        self.position = position.float()
        self.mean = self.anchor
        self.variance = update.initial_variance(self.mean)
        self.last_frame_idx = frame_idx
        self.updates = 0
        self.previous_feature = None  # image features of the previous frame, for the camera-motion measure

    def tensors(self) -> list[torch.Tensor]:
        return [self.anchor, self.position, self.mean, self.variance]


class KalmanMemoryMixin:
    """Put before a ``SAM2Base`` subclass; inactive while ``memory_update`` is None or ``kalman_enabled`` is False."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_update = None
        self.kalman_enabled = True
        self.kalman_last_step = None
        self._kalman_step = None
        # Treat an absent frame as a missing measurement: K = 0 and no object pointer in attention.
        self.kalman_skip_absent = False

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
        step["feature_change"] = change  # camera motion, logged for the motion-stratified analysis
        step["prior_variance"] = self.memory_update.predict(state.variance)
        dtype = current_vision_feats[-1].dtype
        memory_chunks, position_chunks = kalman_memory_tokens(
            self, state.anchor, self.memory_update.readout(state.mean, step["prior_variance"]),
            state.position, dtype,
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
            )
            output_dict["kalman_state"].previous_feature = vision_feature_map(current_vision_feats, feat_sizes).float()
            return
        state = output_dict["kalman_state"]
        presence = torch.sigmoid(object_score_logits.float())
        status = "absent" if self.kalman_skip_absent and bool((presence <= 0.5).all()) else "accepted"
        updated = self.memory_update.update(
            state.mean,
            step["prior_variance"],
            candidate.float(),
            step["projection"],
            torch.sigmoid(current_out["pred_masks"].float()),
            presence,
        )
        if status != "accepted":  # missing measurement: keep the mean, let the variance grow
            updated.update(mean=state.mean, variance=step["prior_variance"], gain=torch.zeros_like(updated["gain"]))
        if not (torch.isfinite(updated["mean"]).all() and torch.isfinite(updated["variance"]).all()):
            raise FloatingPointError(f"Kalman state became non-finite at frame {frame_idx}.")
        state.mean, state.variance = updated["mean"], updated["variance"]
        state.last_frame_idx = frame_idx
        state.updates += 1
        updated.update(presence=presence, prior_variance=step["prior_variance"], status=status)
        if step.get("feature_change") is not None:
            updated["feature_change"] = step["feature_change"]
        current_out["kalman"] = updated
        current_out["kalman_missing"] = status == "absent"
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
        )
        _, _, vision_feats, _, feat_sizes = self._get_image_feature(inference_state, anchor_idx, 1)
        outputs["kalman_state"].previous_feature = vision_feature_map(vision_feats, feat_sizes).float()
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
            current_out["kalman_missing"] = step["status"] == "absent"
            current_out["kalman_trace"] = {
                "kalman_status": step["status"],
                "presence": float(step["presence"].mean()),
                "gain_mean": float(step["gain"].mean()),
                "prior_variance_mean": float(step["prior_variance"].mean()),
                "feature_change_mean": float(step["feature_change"].mean()) if "feature_change" in step else float("nan"),
                "variance_mean": float(step["variance"].mean()),
                "observation_noise_mean": float(step["observation_noise"].mean()),
            }
        return current_out, pred_masks


def kalman_memory_tokens(model, anchor, state_readout, position, dtype=None):
    dtype = dtype or anchor.dtype
    anchor_position = position + model.maskmem_tpos_enc[model.num_maskmem - 1].view(1, -1, 1, 1)
    state_position = position + model.maskmem_tpos_enc[0].view(1, -1, 1, 1)
    tokens = lambda tensor: tensor.to(dtype).flatten(2).permute(2, 0, 1)
    return [tokens(anchor), tokens(state_readout)], [tokens(anchor_position), tokens(state_position)]


def save_memory_update(update, path) -> None:
    """Save a Kalman (``KalmanMemoryUpdate``) or RDE (``RDEMemoryUpdate``) memory update."""
    from modeling.rde_memory import CHECKPOINT_KIND as RDE_KIND, RDEMemoryUpdate

    torch.save({
        "format": CHECKPOINT_FORMAT,
        "kind": RDE_KIND if isinstance(update, RDEMemoryUpdate) else "kalman",
        "memory_channels": update.memory_channels,
        "image_channels": update.image_channels,
        "update_config": update.config,
        "state_dict": update.state_dict(),
    }, path)


def load_memory_update(path, model, device):
    """Memory update from ``save_memory_update``, checked against ``model``'s MedSAM2 widths."""
    from modeling.rde_memory import CHECKPOINT_KIND as RDE_KIND, RDEMemoryUpdate

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in (CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT):
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint.")
    if (checkpoint["memory_channels"], checkpoint["image_channels"]) != (model.mem_dim, model.hidden_dim):
        raise ValueError(f"{path} does not match this MedSAM2 model.")
    kind = RDEMemoryUpdate if checkpoint.get("kind") == RDE_KIND else KalmanMemoryUpdate
    update = kind(model.mem_dim, model.hidden_dim, **checkpoint["update_config"]).to(device)
    state_dict = dict(checkpoint["state_dict"])
    state_dict.pop("detection_trust.weight", None)
    state_dict.pop("detection_trust.bias", None)
    update.load_state_dict(state_dict)
    return update.eval()
