"""Training-free recurrent spatial memory for SAM2 video predictor.
   After the current has predicted the mask.
   How the recurrent memory is represented, read, aligned, updated
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.modeling.sam2_utils import get_1d_sine_pe
from modeling.reliability_gate import (
    reliability_gated_ema_weight,
    token_write_reliability,
)


UPDATE_MODE_COMPONENTS = {
    "direct": ("single", "replace"),
    "fixed": ("single", "ema"),
    "adaptive": ("single", "reliability_gated_ema"),
    "three_timescale": ("three_timescale", "reliability_gated_ema"),
}


def update_mode_components(mode):
    """Return pair: memory representation + write policy for an ablation mode."""
    try:
        return UPDATE_MODE_COMPONENTS[mode]
    except KeyError as error:
        raise ValueError(f"Unknown dynamic-token update mode: {mode}") from error


def warp_token_grid(features, previous_bgr, current_bgr):
    """Estimate flow from images => warp previous spatial memory features - using Farneback algo
       Args: maskmem_features: encoded features from memory encoder, gray-scale images of previous and current frames
       Returns:
       - aligned: previous frame's memory grid warped into the current frame's coordinate
       - valid: binary spatial map of shape
          - 1: warped location sampled from inside the previous grid
          - 0: flow-mapped source outside the grid"""
    # before fusion
    # Ablated through --motion_alignment none vs flow.
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
    """Describe the current mask prediction, expressed at memory-token resolution
       Returns: tensor [B, 1, feature_height, feature_width]"""
    probabilities = mask_logits.detach().to(device=device, dtype=torch.float32).sigmoid()
    return F.interpolate(probabilities, size=feature_size, mode="bilinear", align_corners=False) # resize smoothly to maskmem_fea resolution


def pool_foreground_features(features, foreground_probabilities):
    """summarizes all foreground memory features into 1 object -  create a compact appearance representation
       Args:
       - features [B, C (channels), feature_width, feature_height]: current_outp["maskmem_features"] - current frame's output from SAM2's memory encoder
       - foreground_probabilities [B, 1, feature_width, feature_height]: return of foreground_token_probabilities function
       Returns: [B, C] - ground-weighted feature vector - summarize how strongly the feature dimension appear over foreground object
    """
    if features.shape[0] != foreground_probabilities.shape[0]:
        raise ValueError("Feature and foreground-mask batches must match.")
    weights = foreground_probabilities.to(device=features.device, dtype=features.dtype)
    normalizer = weights.sum(dim=(-2, -1)).clamp_min(torch.finfo(features.dtype).eps) # sum foreground weights
    return (features * weights).sum(dim=(-2, -1)) / normalizer


def token_identity_similarity(foreground_feature, identity_prototype):
    """Cosine consistency of a foreground feature and the recurrent prototype.
       Args: 
       - foreground_feature: current-frame maskmem_features -> pooled vector
       - identity_prototype: return of SAM2's memory-encoder features
    """
    if foreground_feature.shape != identity_prototype.shape:
        raise ValueError("Foreground feature and identity prototype shapes must match.")
    similarity = F.cosine_similarity(
        foreground_feature.float(), identity_prototype.float(), dim=1
    )
    return float(similarity.clamp(0.0, 1.0).mean().item())


class DynamicTokenState:
    """recurrent state for one tracked object across a video"""

    def __init__(
        self,
        seed_frame_idx, # index of the prompt frame
        output, # SAM2 output for the current frame
        frame_bgr, # raw prompt-frame image - future optical flow alignment
        short_read_weight, # weights for reading 3-timscale memory
        reliable_read_weight, # weights for reading 3-timscale memory
        representation, # "single" or "three_timescale"
    ):
        if not 0.0 <= short_read_weight <= 1.0 or not 0.0 <= reliable_read_weight <= 1.0:
            raise ValueError("Read weights must be in [0, 1].")
        if short_read_weight + reliable_read_weight > 1.0:
            raise ValueError("short_read_weight + reliable_read_weight must not exceed 1.")
        if representation not in ("single", "three_timescale"):
            raise ValueError(f"Unknown dynamic-token representation: {representation}")
        self.seed_frame_idx = seed_frame_idx
        self.representation = representation
        # The prompted frame initializes the state, but is not kept as a
        # separate spatial-memory slot.
        device = output["obj_ptr"].device
        initial_features = output["maskmem_features"].detach().to(device=device, dtype=torch.float32) # return of memory encoder
        self.position = output["maskmem_pos_enc"][-1].detach().to(device=device, dtype=torch.float32).clone() # maskmem + pos_enc + memory tokens => attention
        if representation == "single":
            self.single_features = initial_features.clone()
        else: # three-timescale
            self.short_features = initial_features.clone()
            self.reliable_features = initial_features.clone()
            self.unreliable_features = initial_features.clone()
        self.short_read_weight = short_read_weight
        self.reliable_read_weight = reliable_read_weight
        self.previous_frame_bgr = frame_bgr
        # valid: decide whether old memory can be trusted at each spatial location
        self.valid = torch.ones_like(initial_features[:, :1]) # every spatial memory is valid before any optical flow warp
        foreground_probabilities = foreground_token_probabilities(
            output["pred_masks"], initial_features.shape[-2:], device
        )
        self.identity_prototype = pool_foreground_features(self.features, foreground_probabilities)
        if representation == "single":
            self.single_foreground_probabilities = foreground_probabilities.clone()
        else:
            self.short_foreground_probabilities = foreground_probabilities.clone()
            self.reliable_foreground_probabilities = foreground_probabilities.clone()
            self.unreliable_foreground_probabilities = foreground_probabilities.clone()
        # Keep object pointers separate from the recurrent spatial summary.  The
        # seed is the sole conditioning pointer; later decoder pointers are raw
        # non-conditioning history, selected at read time like native SAM2.
        self.conditioning_pointer = output["obj_ptr"].detach().to(device=device).clone()
        self.pointer_history = {}
        self.updates = 0
        self.last_frame_idx = seed_frame_idx

    @property
    def features(self):
        """Returns: spatial memory grid currently used for reading"""
        if self.representation == "single":
            return self.single_features
        # three time-scale
        return (
            self.short_read_weight * self.short_features # change quickly with recent frames
            + self.reliable_read_weight * self.reliable_features # update slowly from reliable predictions
            + (1.0 - self.short_read_weight - self.reliable_read_weight) * self.unreliable_features # retain a separate slow history for uncertain predictions
        )

    @property
    def foreground_probabilities(self):
        if self.representation == "single":
            return self.single_foreground_probabilities
        # blend 3 stored maps using read weights as feature memory
        return (
            self.short_read_weight * self.short_foreground_probabilities
            + self.reliable_read_weight * self.reliable_foreground_probabilities
            + (1.0 - self.short_read_weight - self.reliable_read_weight)
            * self.unreliable_foreground_probabilities
        )

    def pointer_entries_for_read(self, frame_idx, max_pointers):
        """Return the conditioning pointer plus recent raw non-conditioning pointers."""
        if max_pointers < 1 or frame_idx <= self.seed_frame_idx:
            return []
        entries = [(self.seed_frame_idx, self.conditioning_pointer)]
        for time_difference in range(1, max_pointers):
            pointer_frame_idx = frame_idx - time_difference
            pointer = self.pointer_history.get(pointer_frame_idx)
            if pointer is not None:
                entries.append((pointer_frame_idx, pointer))
        return entries

    def align(self, frame_bgr, use_flow):
        """warp the stored spatial meory into current frame's coordinate using optical flow"""
        if use_flow:
            if self.representation == "single":
                aligned, self.valid = warp_token_grid(
                    torch.cat((self.single_features, self.single_foreground_probabilities), dim=1),
                    self.previous_frame_bgr,
                    frame_bgr,
                )
                channels = self.single_features.size(1)
                self.single_features = aligned[:, :channels]
                self.single_foreground_probabilities = aligned[:, channels:]
                return
            aligned, self.valid = warp_token_grid(
                torch.cat((
                    self.short_features,
                    self.reliable_features,
                    self.unreliable_features,
                    self.short_foreground_probabilities,
                    self.reliable_foreground_probabilities,
                    self.unreliable_foreground_probabilities,
                ), dim=1),
                self.previous_frame_bgr, frame_bgr,
            )
            channels = self.short_features.size(1)
            self.short_features = aligned[:, :channels]
            self.reliable_features = aligned[:, channels:2 * channels]
            self.unreliable_features = aligned[:, 2 * channels:3 * channels]
            self.short_foreground_probabilities = aligned[:, 3 * channels:3 * channels + 1]
            self.reliable_foreground_probabilities = aligned[:, 3 * channels + 1:3 * channels + 2]
            self.unreliable_foreground_probabilities = aligned[:, 3 * channels + 2:]
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
        write_policy="reliability_gated_ema",
        reliable_write_rate=0.08,
        unreliable_write_rate=0.02,
        long_term_split_power=1.5,
    ):
        """Take the current's output and write into stored state"""
        # check the update
        if frame_idx != self.last_frame_idx + 1:
            raise ValueError("Token state requires consecutive, forward frame updates.")
        if not np.isfinite(write_weight) or not 0 <= write_weight <= 1:
            raise ValueError("write_weight must be finite in [0, 1].")
        if not 0.0 <= reliable_write_rate <= 1.0:
            raise ValueError("reliable_write_rate must be in [0, 1].")
        if not 0.0 <= unreliable_write_rate <= 1.0:
            raise ValueError("unreliable_write_rate must be in [0, 1].")
        if long_term_split_power <= 0.0:
            raise ValueError("long_term_split_power must be positive.")
        # Native SAM2 retains raw decoder pointers independently of the spatial
        # memory write policy.  Keep every non-conditioning frame available for
        # the native-equivalent recent-pointer selection at read time.
        self.pointer_history[frame_idx] = output["obj_ptr"].detach().to(
            device=self.features.device
        ).clone()
        candidate = output["maskmem_features"].detach().to(self.features)
        if candidate.shape != self.features.shape:
            raise ValueError("Memory encoder feature shape changed within the sequence.")
        probabilities = foreground_probabilities.to(self.features)
        if write_policy == "replace":
            token_weight = torch.ones_like(probabilities)
        elif write_policy == "ema":
            token_weight = torch.full_like(probabilities, write_weight)
        elif write_policy == "reliability_gated_ema":
            # Confident background must also clear stale foreground memory.
            token_weight = write_weight * (2.0 * probabilities - 1.0).abs()
        else:
            raise ValueError(f"Unknown dynamic-token write policy: {write_policy}")

        # Where flow has no source coordinate, use the current candidate as the base.
        short_weight = token_weight
        if self.representation == "single":
            feature_base = self.single_features * self.valid + candidate * (1.0 - self.valid)
            probability_base = (
                self.single_foreground_probabilities * self.valid
                + probabilities * (1.0 - self.valid)
            )
            self.single_features = (1.0 - short_weight) * feature_base + short_weight * candidate
            self.single_foreground_probabilities = (
                (1.0 - short_weight) * probability_base + short_weight * probabilities
            )
            identity_weight = float(short_weight.mean().item())
            reliable_weight = None
            unreliable_weight = None
        else:
            reliable_score = reliability ** long_term_split_power
            unreliable_score = (1.0 - reliability) ** long_term_split_power
            score_sum = max(reliable_score + unreliable_score, 1e-12)
            reliable_fraction = reliable_score / score_sum
            unreliable_fraction = unreliable_score / score_sum
            reliable_weight = (short_weight * reliable_fraction).clamp(max=reliable_write_rate)
            unreliable_weight = (short_weight * unreliable_fraction).clamp(max=unreliable_write_rate)
            short_base = self.short_features * self.valid + candidate * (1.0 - self.valid)
            reliable_base = self.reliable_features * self.valid + candidate * (1.0 - self.valid)
            unreliable_base = self.unreliable_features * self.valid + candidate * (1.0 - self.valid)
            short_probability_base = (
                self.short_foreground_probabilities * self.valid
                + probabilities * (1.0 - self.valid)
            )
            reliable_probability_base = (
                self.reliable_foreground_probabilities * self.valid
                + probabilities * (1.0 - self.valid)
            )
            unreliable_probability_base = (
                self.unreliable_foreground_probabilities * self.valid
                + probabilities * (1.0 - self.valid)
            )
            self.short_features = (1.0 - short_weight) * short_base + short_weight * candidate
            self.reliable_features = (
                (1.0 - reliable_weight) * reliable_base + reliable_weight * candidate
            )
            self.unreliable_features = (
                (1.0 - unreliable_weight) * unreliable_base + unreliable_weight * candidate
            )
            self.short_foreground_probabilities = (
                (1.0 - short_weight) * short_probability_base
                + short_weight * probabilities
            )
            self.reliable_foreground_probabilities = (
                (1.0 - reliable_weight) * reliable_probability_base
                + reliable_weight * probabilities
            )
            self.unreliable_foreground_probabilities = (
                (1.0 - unreliable_weight) * unreliable_probability_base
                + unreliable_weight * probabilities
            )
            identity_weight = float(reliable_weight.mean().item())
        if has_foreground and identity_weight > 0.0:
            self.identity_prototype = (
                (1.0 - identity_weight) * self.identity_prototype
                + identity_weight * foreground_feature.to(self.identity_prototype)
            )
        self.previous_frame_bgr = frame_bgr
        self.last_frame_idx = frame_idx
        self.updates += 1
        return short_weight, reliable_weight, unreliable_weight


class DynamicTokenVideoPredictor(SAM2VideoPredictor):
    """Reuse native decode/encode; replace only construction of memory attention inputs.

    Only forward, single-object inference with one initial prompt is supported.
    The inference script owns/reset the explicit DynamicTokenState per sequence.
    """

    def _new_token_state(self, inference_state, frame_idx, output):
        if output["maskmem_features"] is None:
            raise RuntimeError("Dynamic-token state requires encoded memory features.")
        representation, _ = update_mode_components(
            inference_state["dynamic_token_update_mode"]
        )
        return DynamicTokenState(
            frame_idx,
            output,
            inference_state["dynamic_token_frames_bgr"][frame_idx],
            inference_state["dynamic_token_short_read_weight"],
            inference_state["dynamic_token_reliable_read_weight"],
            representation,
        )

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
        if prompted_output is None:
            raise RuntimeError("VOS preflight did not encode the prompted frame.")
        inference_state["output_dict"]["token_state"] = self._new_token_state(
            inference_state, frame_idx, prompted_output
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
        reliability = inference_state.get("dynamic_token_fixed_reliability")
        if reliability is None:
            reliability = token_write_reliability(
                mask_confidence,
                object_probability,
                has_foreground,
                identity_similarity,
                inference_state["dynamic_token_unconfirmed_absence_scale"],
            )
        representation, write_policy = update_mode_components(
            inference_state["dynamic_token_update_mode"]
        )
        if token_state.representation != representation:
            raise RuntimeError("Dynamic-token representation changed during propagation.")
        if write_policy == "replace":
            write_weight = 1.0
        elif write_policy == "ema":
            write_weight = inference_state["dynamic_token_fixed_weight"]
        else:
            write_weight = reliability_gated_ema_weight(
                reliability,
                inference_state["dynamic_token_write_rate"],
                inference_state["dynamic_token_max_smooth_write"],
                inference_state["dynamic_token_reliability_power"],
            )
        background_write_capped = False
        if not has_foreground and write_policy == "reliability_gated_ema":
            capped_weight = min(
                write_weight, inference_state["dynamic_token_max_background_write"]
            )
            background_write_capped = capped_weight < write_weight
            write_weight = capped_weight
        short_weights, reliable_weights, unreliable_weights = token_state.update(
            frame_idx,
            current_out,
            frame_bgr,
            write_weight,
            reliability,
            has_foreground,
            foreground_probabilities,
            foreground_feature,
            write_policy,
            reliable_write_rate=inference_state["dynamic_token_reliable_write_rate"],
            unreliable_write_rate=inference_state["dynamic_token_unreliable_write_rate"],
            long_term_split_power=inference_state["dynamic_token_long_term_split_power"],
        )
        # for savings
        current_out["dynamic_token_trace"] = {
            "predicted_iou": mask_confidence,
            "object_probability": object_probability,
            "identity_similarity": identity_similarity,
            "foreground_fraction": float((foreground_probabilities > 0.5).float().mean().item()),
            "reliability": reliability,
            "memory_representation": representation,
            "write_policy": write_policy,
            "write_weight": write_weight,
            "has_foreground": has_foreground,
            "background_write_capped": background_write_capped,
            "mean_short_token_weight": float(short_weights.mean().item()),
            "mean_reliable_token_weight": (
                None if reliable_weights is None else float(reliable_weights.mean().item())
            ),
            "mean_unreliable_token_weight": (
                None if unreliable_weights is None else float(unreliable_weights.mean().item())
            ),
            "min_short_token_weight": float(short_weights.min().item()),
            "max_short_token_weight": float(short_weights.max().item()),
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
        if self.use_obj_ptrs_in_encoder:
            limit = min(num_frames, self.max_obj_ptrs_in_encoder)
            pointer_entries = state.pointer_entries_for_read(frame_idx, limit)
            if pointer_entries:
                pointer_indices, pointer_values = zip(*pointer_entries)
                pointer_indices = list(pointer_indices)
                pointers = torch.stack(pointer_values, dim=0).to(dtype)
                if self.add_tpos_enc_to_obj_ptrs:
                    distances = torch.tensor(
                        [frame_idx - pointer_idx for pointer_idx in pointer_indices],
                        device=pointers.device,
                        dtype=torch.float32,
                    )
                    dim = self.hidden_dim if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                    pointer_pos = get_1d_sine_pe(distances / max(limit - 1, 1), dim=dim)
                    pointer_pos = self.obj_ptr_tpos_proj(pointer_pos.to(dtype)).unsqueeze(1)
                else:
                    pointer_pos = pointers.new_zeros(len(pointer_entries), 1, self.mem_dim)
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
