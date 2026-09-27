"""Compact recurrent spatial memory for MedSAM2 video inference."""

import torch
import torch.nn.functional as F

from MedSAM2.sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames
from sam2.sam2_video_predictor import SAM2VideoPredictor

# Archived compact-memory ablation backends. Native MedSAM2 is the active
# baseline; this module is retained so the negative result remains reproducible.
MEMORY_BACKENDS = ("native", "fixed_ema", "adaptive")


def append_native_object_pointers(
    model, frame_idx, output_dict, num_frames, track_in_reverse, device, batch_size,
    memory_chunks, position_chunks,
):
    """Append MedSAM2's normal object-pointer tokens to recurrent spatial memory."""
    if not model.use_obj_ptrs_in_encoder:
        return 0
    selected_cond, unselected_cond = select_closest_cond_frames(
        frame_idx, output_dict["cond_frame_outputs"], model.max_cond_frames_in_attn
    )
    if not model.training and model.only_obj_ptrs_in_the_past_for_eval:
        pointer_conditions = {
            t: output for t, output in selected_cond.items()
            if (t >= frame_idx if track_in_reverse else t <= frame_idx)
        }
    else:
        pointer_conditions = selected_cond
    sign = -1 if track_in_reverse else 1
    positions_and_pointers = [
        ((frame_idx - t) * sign if model.use_signed_tpos_enc_to_obj_ptrs else abs(frame_idx - t), output["obj_ptr"])
        for t, output in pointer_conditions.items()
    ]
    max_pointers = min(num_frames, model.max_obj_ptrs_in_encoder)
    for frame_distance in range(1, max_pointers):
        t = frame_idx + frame_distance if track_in_reverse else frame_idx - frame_distance
        if t < 0 or (num_frames is not None and t >= num_frames):
            break
        output = output_dict["non_cond_frame_outputs"].get(t, unselected_cond.get(t))
        if output is not None:
            positions_and_pointers.append((frame_distance, output["obj_ptr"]))
    if not positions_and_pointers:
        return 0
    positions, pointers = zip(*positions_and_pointers)
    pointers = torch.stack(pointers, dim=0).to(device)
    if model.add_tpos_enc_to_obj_ptrs:
        position_dim = model.hidden_dim if model.proj_tpos_enc_in_obj_ptrs else model.mem_dim
        temporal_positions = get_1d_sine_pe(
            torch.tensor(positions, device=device) / (max_pointers - 1), dim=position_dim
        )
        pointer_positions = model.obj_ptr_tpos_proj(temporal_positions)
        pointer_positions = pointer_positions.unsqueeze(1).expand(-1, batch_size, model.mem_dim)
    else:
        pointer_positions = pointers.new_zeros(len(positions), batch_size, model.mem_dim)
    if model.mem_dim < model.hidden_dim:
        pointers = pointers.reshape(-1, batch_size, model.hidden_dim // model.mem_dim, model.mem_dim)
        pointers = pointers.permute(0, 2, 1, 3).flatten(0, 1)
        pointer_positions = pointer_positions.repeat_interleave(model.hidden_dim // model.mem_dim, dim=0)
    memory_chunks.append(pointers)
    position_chunks.append(pointer_positions)
    return pointers.shape[0]


class DynamicMemoryState:
    """One recurrent spatial memory state for a single prompted object."""

    def __init__(self, frame_idx, output):
        if output["maskmem_features"] is None:
            raise RuntimeError("Dynamic-state inference requires memory features.")
        device = output["obj_ptr"].device
        self.features = output["maskmem_features"].detach().to(
            device=device, dtype=torch.float32
        )
        self.position = output["maskmem_pos_enc"][-1].detach().to(
            device=device, dtype=torch.float32
        )
        self.last_frame_idx = frame_idx
        self.updates = 0

    def _check_candidate(self, frame_idx, candidate):
        if frame_idx != self.last_frame_idx + 1:
            raise ValueError("Dynamic state requires consecutive forward updates.")
        candidate = candidate.to(self.features)
        if candidate.shape != self.features.shape:
            raise ValueError("Memory feature shape changed within the sequence.")
        return candidate

    def update_fixed(self, frame_idx, candidate, weight):
        """Update with a fixed scalar EMA weight (ablation baseline)."""
        candidate = self._check_candidate(frame_idx, candidate).detach()
        weight = weight.view(-1, 1, 1, 1).to(self.features)
        self.features = (1.0 - weight) * self.features + weight * candidate
        self.last_frame_idx = frame_idx
        self.updates += 1

    def update_adaptive(self, frame_idx, candidate, foreground_probability, fusion):
        """Update with a learned spatial/channel gate."""
        candidate = self._check_candidate(frame_idx, candidate).detach()
        previous = self.features
        probability = foreground_probability.to(previous)
        self.features = fusion(previous, candidate, probability)
        self.last_frame_idx = frame_idx
        self.updates += 1


class CompactStateVideoPredictor(SAM2VideoPredictor):
    """MedSAM2 predictor using one recurrent spatial memory state."""

    def __init__(
        self,
        state_update_mode="fixed_ema",
        fixed_ema_alpha=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if state_update_mode not in {"fixed_ema", "adaptive"}:
            raise ValueError(f"Unknown dynamic-state update mode: {state_update_mode}")
        if state_update_mode == "fixed_ema" and not 0.0 <= fixed_ema_alpha <= 1.0:
            raise ValueError("fixed_ema_alpha must be in [0, 1].")
        self.state_update_mode = state_update_mode
        self.fixed_ema_alpha = fixed_ema_alpha
        # Attach trained fusion weights after MedSAM2's base checkpoint loads.
        # Registering this new module here would make upstream strict checkpoint
        # loading expect fusion parameters that do not exist in MedSAM2 weights.
        self.state_fusion = None

    def propagate_in_video_preflight(self, inference_state):
        if not inference_state.get("compact_state_enabled", False):
            raise ValueError("Dynamic-state inference must be enabled before propagation.")
        if self._get_obj_num(inference_state) != 1:
            raise ValueError("Dynamic-state inference currently supports exactly one object.")
        super().propagate_in_video_preflight(inference_state)
        frame_idx = inference_state["compact_state_anchor_frame_idx"]
        if set(inference_state["output_dict"]["cond_frame_outputs"]) != {frame_idx}:
            raise ValueError("Dynamic-state inference supports one initial prompt frame.")
        output_dict = inference_state["output_dict"]
        if "compact_state" not in output_dict:
            output = output_dict["cond_frame_outputs"].get(frame_idx)
            if output is None:
                raise RuntimeError("VOS preflight did not encode the prompted frame.")
            output_dict["compact_state"] = DynamicMemoryState(frame_idx, output)
            # Keep the copied recurrent state, not a duplicate prompt-frame
            # spatial memory tensor in the native conditioning dictionary.
            output["maskmem_features"] = None
            output["maskmem_pos_enc"] = None

    def _reset_tracking_results(self, inference_state):
        super()._reset_tracking_results(inference_state)
        inference_state["output_dict"].pop("compact_state", None)

    def _run_single_frame_inference(self, *args, **kwargs):
        output_dict = kwargs["output_dict"]
        frame_idx = kwargs["frame_idx"]
        state = output_dict.get("compact_state")
        if state is None or kwargs["is_init_cond_frame"]:
            return super()._run_single_frame_inference(*args, **kwargs)
        if kwargs["reverse"]:
            raise ValueError("Dynamic-state inference supports forward propagation only.")

        current_out, pred_masks = super()._run_single_frame_inference(*args, **kwargs)
        candidate = current_out["maskmem_features"]
        if candidate is None:
            raise RuntimeError("Dynamic-state inference requires memory features.")

        if self.state_update_mode == "adaptive":
            if self.state_fusion is None:
                raise RuntimeError("Adaptive inference requires trained state-fusion weights.")
            probability = torch.sigmoid(pred_masks.detach())
            probability = F.interpolate(
                probability,
                size=candidate.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            state.update_adaptive(
                frame_idx,
                candidate,
                probability,
                self.state_fusion,
            )
            trace = {
                "state_update_mode": "adaptive",
                "updates": state.updates,
            }
        else:
            weight = torch.full(
                (candidate.size(0),),
                self.fixed_ema_alpha,
                device=state.features.device,
                dtype=state.features.dtype,
            )
            state.update_fixed(frame_idx, candidate, weight)
            trace = {
                "state_update_mode": "fixed_ema",
                "update_weight": float(weight.mean().item()),
                "updates": state.updates,
            }

        # The recurrent state is now the sole spatial memory read by later
        # frames. Retaining this frame's memory-encoder tensors in the native
        # output dictionary would defeat the compact-memory storage claim.
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["compact_state_trace"] = trace
        return current_out, pred_masks

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
        if is_init_cond_frame:
            return super()._prepare_memory_conditioned_features(
                frame_idx,
                True,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                output_dict,
                num_frames,
                track_in_reverse,
            )
        if track_in_reverse:
            raise ValueError("Dynamic-state inference supports forward propagation only.")
        state = output_dict["compact_state"]
        if frame_idx != state.last_frame_idx + 1:
            raise ValueError("Dynamic state was read out of order.")
        dtype = current_vision_feats[-1].dtype
        memory = state.features.to(dtype).flatten(2).permute(2, 0, 1)
        memory_position = (
            state.position + self.maskmem_tpos_enc[0].view(1, -1, 1, 1)
        ).to(dtype).flatten(2).permute(2, 0, 1)
        memory_chunks = [memory]
        position_chunks = [memory_position]
        num_obj_ptr_tokens = append_native_object_pointers(
            self,
            frame_idx,
            output_dict,
            num_frames,
            track_in_reverse,
            current_vision_feats[-1].device,
            current_vision_feats[-1].size(1),
            memory_chunks,
            position_chunks,
        )
        fused = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=torch.cat(memory_chunks, dim=0),
            memory_pos=torch.cat(position_chunks, dim=0),
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        batch_size = current_vision_feats[-1].size(1)
        return fused.permute(1, 2, 0).reshape(
            batch_size, self.hidden_dim, *feat_sizes[-1]
        )
