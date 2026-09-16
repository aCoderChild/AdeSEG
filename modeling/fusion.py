"""Training-free recurrent spatial memory for this repository's SAM2 predictor.

One evolving BCHW memory grid replaces the native spatial queue. It is seeded
from the prompted frame, then becomes a lossy exponential summary of history.
Only pointers from propagated frames are read. No new learned parameters are
introduced; compatibility with the frozen attention is empirical.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.modeling.sam2_utils import get_1d_sine_pe
from modeling.reliability_gate import token_write_reliability


MAX_SMOOTH_WRITE = 0.70
SMOOTH_RELIABILITY_POWER = 2.0


def warp_token_grid(features, previous_bgr, current_bgr):
    """Backward optical flow resamples old features into current coordinates.

    Flow is estimated at 128x128, then scaled to the memory grid. Position
    embeddings are NOT warped or averaged: the result lives on the current grid.
    Returns the resampled features and validity of their source coordinates.
    """
    # moves previous token-state feature map -> coordinates of the current video frame before fusion
    # TODO: ablation: --motion-alignment none vs flow
    if previous_bgr is None:
        return features, torch.ones_like(features[:, :1])
    gray = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (128, 128))
            for x in (current_bgr, previous_bgr)]
    flow = cv2.calcOpticalFlowFarneback(gray[0], gray[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    height, width = features.shape[-2:]
    flow = cv2.resize(flow, (width, height))
    flow *= np.array([width / 128, height / 128], dtype=np.float32)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    x, y = xx + flow[..., 0], yy + flow[..., 1]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    grid = np.stack((2 * (x + 0.5) / width - 1, 2 * (y + 0.5) / height - 1), axis=-1)
    grid = torch.as_tensor(grid, device=features.device)[None].expand(features.size(0), -1, -1, -1)
    aligned = F.grid_sample(features.float(), grid, align_corners=False, padding_mode="zeros")
    valid = torch.as_tensor(valid, device=features.device, dtype=torch.float32)[None, None]
    return aligned, valid


def foreground_token_probabilities(mask_logits, feature_size, device):
    """Map a decoded mask to soft foreground weights on the memory-token grid."""
    probabilities = mask_logits.detach().to(device=device, dtype=torch.float32).sigmoid()
    return F.interpolate(probabilities, size=feature_size, mode="bilinear", align_corners=False)


def pool_foreground_features(features, foreground_probabilities):
    """Pool memory features using soft foreground weights."""
    if features.shape[0] != foreground_probabilities.shape[0]:
        raise ValueError("Feature and foreground-mask batches must match.")
    weights = foreground_probabilities.to(device=features.device, dtype=features.dtype)
    normalizer = weights.sum(dim=(-2, -1)).clamp_min(torch.finfo(features.dtype).eps)
    return (features * weights).sum(dim=(-2, -1)) / normalizer


def token_identity_similarity(foreground_feature, identity_prototype):
    """Cosine consistency of a foreground feature and the recurrent prototype."""
    if foreground_feature.shape != identity_prototype.shape:
        raise ValueError("Foreground feature and identity prototype shapes must match.")
    similarity = F.cosine_similarity(
        foreground_feature.float(), identity_prototype.float(), dim=1
    )
    return float(similarity.clamp(0.0, 1.0).mean().item())


def area_plausibility(current_foreground, reference_foreground):
    """Score current foreground area against the aligned state."""
    current_area = float(current_foreground.mean().item())
    reference_area = float(reference_foreground.mean().item())
    if current_area == 0.0 or reference_area == 0.0:
        return 1.0
    return (min(current_area, reference_area) / max(current_area, reference_area)) ** 0.5


def soft_foreground_iou(current_foreground, reference_foreground):
    """Measure token-grid foreground agreement."""
    intersection = torch.minimum(current_foreground, reference_foreground).sum()
    union = torch.maximum(current_foreground, reference_foreground).sum()
    if float(union.item()) == 0.0:
        return 1.0
    return float((intersection / union).item())


def foreground_consistency_map(current_foreground, reference_foreground):
    """Return local agreement between current and aligned foreground maps."""
    return 1.0 - (current_foreground - reference_foreground).abs()


def smooth_write_weight(reliability, minimum_weight):
    """Map reliability to a continuous recurrent-state update rate."""
    if not 0.0 <= reliability <= 1.0:
        raise ValueError("reliability must be in [0, 1].")
    if not 0.0 <= minimum_weight <= MAX_SMOOTH_WRITE:
        raise ValueError("minimum_weight must be in [0, MAX_SMOOTH_WRITE].")
    return minimum_weight + (MAX_SMOOTH_WRITE - minimum_weight) * (
        reliability ** SMOOTH_RELIABILITY_POWER
    )


class DynamicTokenState:
    """Sequence-local state. Update AFTER decoding; read only past observations."""

    def __init__(self, seed_frame_idx, output, frame_bgr):
        self.seed_frame_idx = seed_frame_idx
        # The prompted frame initializes the state, but is not kept as a
        # separate spatial-memory slot.
        device = output["obj_ptr"].device
        initial_features = output["maskmem_features"].detach().to(device=device, dtype=torch.float32)
        self.position = output["maskmem_pos_enc"][-1].detach().to(device=device, dtype=torch.float32).clone()
        self.features = initial_features.clone()
        self.previous_frame_bgr = frame_bgr
        self.valid = torch.ones_like(self.features[:, :1])
        foreground_probabilities = foreground_token_probabilities(
            output["pred_masks"], self.features.shape[-2:], device
        )
        self.identity_prototype = pool_foreground_features(
            self.features, foreground_probabilities
        )
        self.foreground_probabilities = foreground_probabilities
        self.pointer = None
        self.pointer_frame_idx = None
        self.updates = 0
        self.last_frame_idx = seed_frame_idx

    def align(self, frame_bgr, use_flow):
        if use_flow:
            aligned, self.valid = warp_token_grid(
                torch.cat((self.features, self.foreground_probabilities), dim=1),
                self.previous_frame_bgr, frame_bgr,
            )
            self.features, self.foreground_probabilities = aligned[:, :-1], aligned[:, -1:]
        else:
            self.valid = torch.ones_like(self.features[:, :1])

    def update(
        self,
        frame_idx,
        output,
        frame_bgr,
        write_weight,
        reliability,
        has_foreground,
        foreground_probabilities,
        foreground_feature,
        local_consistency,
        update_mode="adaptive",
    ):
        if frame_idx != self.last_frame_idx + 1:
            raise ValueError("Token state requires consecutive, forward frame updates.")
        if not np.isfinite(write_weight) or not 0 <= write_weight <= 1:
            raise ValueError("write_weight must be finite in [0, 1].")
        candidate = output["maskmem_features"].detach().to(self.features)
        if candidate.shape != self.features.shape:
            raise ValueError("Memory encoder feature shape changed within the sequence.")
        probabilities = foreground_probabilities.to(self.features)
        if update_mode == "adaptive":
            # Confident background must also clear stale foreground memory.
            token_weight = (
                write_weight * (2.0 * probabilities - 1.0).abs()
                * (reliability + (1.0 - reliability) * local_consistency.to(self.features))
            )
        elif update_mode in ("direct", "fixed"):
            token_weight = torch.full_like(probabilities, write_weight)
        else:
            raise ValueError(f"Unknown dynamic-token update mode: {update_mode}")
        # Where flow has no source coordinate, only the current feature is usable.
        weight = token_weight * self.valid + (1.0 - self.valid)
        self.features = (1.0 - weight) * self.features + weight * candidate
        self.foreground_probabilities = (
            (1.0 - weight) * self.foreground_probabilities
            + weight * probabilities
        )
        if has_foreground and write_weight > 0.0:
            pointer = output["obj_ptr"].detach().to(self.features.device)
            if self.pointer is None:
                self.pointer = pointer.clone()
            else:
                self.pointer = (1.0 - write_weight) * self.pointer + write_weight * pointer
            self.pointer_frame_idx = frame_idx
            self.identity_prototype = (
                (1.0 - write_weight) * self.identity_prototype
                + write_weight * foreground_feature.to(self.identity_prototype)
            )
        self.previous_frame_bgr = frame_bgr
        self.last_frame_idx = frame_idx
        self.updates += 1
        return weight


class DynamicTokenVideoPredictor(SAM2VideoPredictor):
    """Reuse native decode/encode; replace only construction of memory attention inputs.

    Only forward, single-object inference with one initial prompt is supported.
    The inference script owns/reset the explicit DynamicTokenState per sequence.
    """

    def propagate_in_video_preflight(self, inference_state):
        """Let SAM2 consolidate the prompt, then initialize the dynamic VOS state."""
        if not inference_state.get("dynamic_token_enabled", False):
            raise ValueError("Dynamic-token inference must be enabled before propagation.")
        if self._get_obj_num(inference_state) != 1:
            raise ValueError("Dynamic tokens require exactly one prompted object.")
        super().propagate_in_video_preflight(inference_state)
        frame_idx = inference_state["dynamic_token_anchor_frame_idx"]
        if set(inference_state["output_dict"]["cond_frame_outputs"]) != {frame_idx}:
            raise ValueError("Dynamic tokens support only one initial prompt frame.")
        if "token_state" in inference_state["output_dict"]:
            return  # Preserve the recurrent summary when propagation is resumed.
        prompted_output = inference_state["output_dict"]["cond_frame_outputs"].get(frame_idx)
        if prompted_output is None or prompted_output["maskmem_features"] is None:
            raise RuntimeError("VOS preflight did not encode the prompted frame.")
        frames_bgr = inference_state["dynamic_token_frames_bgr"]
        inference_state["output_dict"]["token_state"] = DynamicTokenState(
            frame_idx,
            prompted_output,
            frames_bgr[frame_idx],
        )

    def _reset_tracking_results(self, inference_state):
        super()._reset_tracking_results(inference_state)
        inference_state["output_dict"].pop("token_state", None)
        inference_state["output_dict"].pop("token_read_trace", None)

    def _run_single_frame_inference(self, *args, **kwargs):
        """Run official VOS decoding and write its result into the token state."""
        output_dict = kwargs["output_dict"]
        frame_idx = kwargs["frame_idx"]
        token_state = output_dict.get("token_state")
        if token_state is None or kwargs["is_init_cond_frame"]:
            return super()._run_single_frame_inference(*args, **kwargs)

        inference_state = kwargs["inference_state"]
        if kwargs["reverse"] or frame_idx != token_state.last_frame_idx + 1:
            raise ValueError("Dynamic tokens require consecutive forward propagation; reset before replay.")
        frame_bgr = inference_state["dynamic_token_frames_bgr"][frame_idx]
        token_state.align(frame_bgr, inference_state["dynamic_token_use_flow"])
        current_out, pred_masks = super()._run_single_frame_inference(*args, **kwargs)

        mask_confidence = float(current_out["iou_predictions"].max(dim=-1).values.item())
        object_probability = float(current_out["object_score_logits"].sigmoid().item())
        has_foreground = bool((pred_masks > 0).any().item())
        foreground_probabilities = foreground_token_probabilities(
            pred_masks, token_state.features.shape[-2:], token_state.features.device
        )
        foreground_feature = pool_foreground_features(
            current_out["maskmem_features"].detach().to(token_state.features),
            foreground_probabilities,
        )
        identity_similarity = token_identity_similarity(
            foreground_feature, token_state.identity_prototype
        )
        area_score = area_plausibility(
            foreground_probabilities, token_state.foreground_probabilities
        )
        temporal_score = soft_foreground_iou(
            foreground_probabilities, token_state.foreground_probabilities
        )
        local_consistency = foreground_consistency_map(
            foreground_probabilities, token_state.foreground_probabilities
        )
        reliability = inference_state.get("dynamic_token_fixed_reliability")
        if reliability is None:
            reliability = token_write_reliability(
                mask_confidence,
                object_probability,
                has_foreground,
                identity_similarity,
                area_score,
                temporal_score,
            )
        mode = inference_state["dynamic_token_update_mode"]
        if mode == "direct":
            write_weight = 1.0
        elif mode == "fixed":
            write_weight = inference_state["dynamic_token_fixed_weight"]
        elif mode == "adaptive":
            write_weight = smooth_write_weight(
                reliability, inference_state["dynamic_token_write_rate"]
            )
        else:
            raise ValueError(f"Unknown dynamic-token update mode: {mode}")
        weights = token_state.update(
            frame_idx,
            current_out,
            frame_bgr,
            write_weight,
            reliability,
            has_foreground,
            foreground_probabilities,
            foreground_feature,
            local_consistency,
            mode,
        )
        current_out["dynamic_token_trace"] = {
            "predicted_iou": mask_confidence,
            "object_probability": object_probability,
            "identity_similarity": identity_similarity,
            "area_plausibility": area_score,
            "temporal_consistency": temporal_score,
            "foreground_fraction": float((foreground_probabilities > 0.5).float().mean().item()),
            "reliability": reliability,
            "write_weight": write_weight,
            "has_foreground": has_foreground,
            "mean_token_weight": float(weights.mean().item()),
            "min_token_weight": float(weights.min().item()),
            "max_token_weight": float(weights.max().item()),
            "flow_invalid_fraction": float((1.0 - token_state.valid).mean().item()),
            "state_shape": list(token_state.features.shape),
            "updates": token_state.updates,
            **output_dict.get("token_read_trace", {}),
        }
        return current_out, pred_masks

    def _prepare_memory_conditioned_features(
        self, frame_idx, is_init_cond_frame, current_vision_feats,
        current_vision_pos_embeds, feat_sizes, output_dict, num_frames,
        track_in_reverse=False,
    ):
        if is_init_cond_frame:
            return super()._prepare_memory_conditioned_features(
                frame_idx, True, current_vision_feats, current_vision_pos_embeds,
                feat_sizes, output_dict, num_frames, track_in_reverse)
        state = output_dict["token_state"]
        if track_in_reverse or self.training or current_vision_feats[-1].size(1) != 1:
            raise ValueError("Dynamic tokens support forward, single-object evaluation only.")
        if frame_idx != state.last_frame_idx + 1:
            raise ValueError("Attempt to read non-causal or out-of-order token state.")
        dtype = current_vision_feats[-1].dtype
        # The prompted frame seeded ``state.features`` but is not retained as a
        # separate spatial slot. The recurrent state always occupies slot 0.
        memories = [state.features]
        positions = [state.position + self.maskmem_tpos_enc[0].view(1, -1, 1, 1)]
        memory = [x.to(dtype).flatten(2).permute(2, 0, 1) for x in memories]
        position = [x.to(dtype).flatten(2).permute(2, 0, 1) for x in positions]
        pointer_indices = []
        pointer_token_count = 0
        if self.use_obj_ptrs_in_encoder and state.pointer is not None:
            limit = min(num_frames, self.max_obj_ptrs_in_encoder)
            pointer_age = frame_idx - state.pointer_frame_idx
            if 0 < pointer_age < limit:
                pointer_indices = [state.pointer_frame_idx]
                pointers = state.pointer.unsqueeze(0).to(dtype)
                if self.add_tpos_enc_to_obj_ptrs:
                    distances = torch.tensor([pointer_age],
                                             device=pointers.device, dtype=torch.float32)
                    dim = self.hidden_dim if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                    pointer_pos = get_1d_sine_pe(distances / max(limit - 1, 1), dim=dim)
                    pointer_pos = self.obj_ptr_tpos_proj(pointer_pos.to(dtype)).unsqueeze(1)
                else:
                    pointer_pos = pointers.new_zeros(1, 1, self.mem_dim)
                if self.mem_dim < self.hidden_dim:
                    splits = self.hidden_dim // self.mem_dim
                    pointers = pointers.reshape(-1, 1, splits, self.mem_dim).permute(0, 2, 1, 3).flatten(0, 1)
                    pointer_pos = pointer_pos.repeat_interleave(splits, dim=0)
                memory.append(pointers)
                position.append(pointer_pos)
                pointer_token_count = pointers.shape[0]
        # Trace describes the INPUT state, before this frame is encoded/written.
        output_dict["token_read_trace"] = {
            "seed_frame_idx": state.seed_frame_idx, "state_through_idx": state.last_frame_idx,
            "spatial_tokens": sum(x.shape[0] for x in memory) - pointer_token_count,
            "pointer_indices": pointer_indices, "pointer_tokens": pointer_token_count,
        }
        fused = self.memory_attention(
            curr=current_vision_feats, curr_pos=current_vision_pos_embeds,
            memory=torch.cat(memory), memory_pos=torch.cat(position),
            num_obj_ptr_tokens=pointer_token_count)
        return fused.permute(1, 2, 0).reshape(1, self.hidden_dim, *feat_sizes[-1])
