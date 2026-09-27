"""Compact fixed-update spatial memory for MedSAM2 video inference."""

import torch

from models.state_memory import append_native_object_pointers
from sam2.sam2_video_predictor import SAM2VideoPredictor


class CompactEMAState:
    """One recurrent spatial memory state for a single prompted object."""

    def __init__(self, frame_idx, output):
        if output["maskmem_features"] is None:
            raise RuntimeError("Compact-state inference requires memory features.")
        device = output["obj_ptr"].device
        self.features = output["maskmem_features"].detach().to(
            device=device, dtype=torch.float32
        )
        self.position = output["maskmem_pos_enc"][-1].detach().to(
            device=device, dtype=torch.float32
        )
        self.last_frame_idx = frame_idx
        self.updates = 0

    def update(self, frame_idx, candidate, weight):
        if frame_idx != self.last_frame_idx + 1:
            raise ValueError("Compact state requires consecutive forward updates.")
        candidate = candidate.detach().to(self.features)
        if candidate.shape != self.features.shape:
            raise ValueError("Memory feature shape changed within the sequence.")
        weight = weight.view(-1, 1, 1, 1).to(self.features)
        self.features = (1.0 - weight) * self.features + weight * candidate
        self.last_frame_idx = frame_idx
        self.updates += 1


class CompactStateVideoPredictor(SAM2VideoPredictor):
    """Predict masks using one spatial state with a fixed update rule."""

    def __init__(
        self,
        state_update_mode="fixed_ema",
        fixed_ema_alpha=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if state_update_mode not in {"current", "fixed_ema"}:
            raise ValueError(f"Unknown compact-state update mode: {state_update_mode}")
        if state_update_mode == "fixed_ema" and not 0.0 <= fixed_ema_alpha <= 1.0:
            raise ValueError("fixed_ema_alpha must be in [0, 1].")
        self.state_update_mode = state_update_mode
        self.fixed_ema_alpha = fixed_ema_alpha

    def propagate_in_video_preflight(self, inference_state):
        if not inference_state.get("compact_state_enabled", False):
            raise ValueError("Compact-state inference must be enabled before propagation.")
        if self._get_obj_num(inference_state) != 1:
            raise ValueError("Compact-state inference supports exactly one object.")
        super().propagate_in_video_preflight(inference_state)
        frame_idx = inference_state["compact_state_anchor_frame_idx"]
        if set(inference_state["output_dict"]["cond_frame_outputs"]) != {frame_idx}:
            raise ValueError("Compact-state inference supports one initial prompt frame.")
        output_dict = inference_state["output_dict"]
        if "compact_state" not in output_dict:
            output = output_dict["cond_frame_outputs"].get(frame_idx)
            if output is None:
                raise RuntimeError("VOS preflight did not encode the prompted frame.")
            output_dict["compact_state"] = CompactEMAState(frame_idx, output)

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
            raise ValueError("Compact-state inference supports forward propagation only.")

        current_out, pred_masks = super()._run_single_frame_inference(*args, **kwargs)
        candidate = current_out["maskmem_features"]
        if candidate is None:
            raise RuntimeError("Compact-state inference requires memory features.")
        weight = (
            torch.ones(candidate.size(0), device=candidate.device, dtype=candidate.dtype)
            if self.state_update_mode == "current"
            else torch.full(
                (candidate.size(0),),
                self.fixed_ema_alpha,
                device=candidate.device,
                dtype=candidate.dtype,
            )
        )
        state.update(frame_idx, candidate, weight)
        current_out["compact_state_trace"] = {
            "state_update_mode": self.state_update_mode,
            "update_weight": float(weight.mean().item()),
            "updates": state.updates,
        }
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
            raise ValueError("Compact-state inference supports forward propagation only.")
        state = output_dict["compact_state"]
        if frame_idx != state.last_frame_idx + 1:
            raise ValueError("Compact state was read out of order.")
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
