"""Learned single-state fusion for MedSAM2 video inference."""

import torch
import torch.nn.functional as F

from modeling.reliability_gate import LearnedReliabilityGate
from modeling.state_memory import append_native_object_pointers
from sam2.sam2_video_predictor import SAM2VideoPredictor


class LearnedState:
    """One recurrent spatial memory state for a single prompted object."""

    def __init__(self, frame_idx, output):
        if output["maskmem_features"] is None:
            raise RuntimeError("Learned-state inference requires memory features.")
        device = output["obj_ptr"].device
        self.features = output["maskmem_features"].detach().to(
            device=device, dtype=torch.float32
        )
        self.position = output["maskmem_pos_enc"][-1].detach().to(
            device=device, dtype=torch.float32
        )
        self.last_frame_idx = frame_idx
        self.updates = 0

    def update(self, frame_idx, candidate, gate):
        if frame_idx != self.last_frame_idx + 1:
            raise ValueError("Learned state requires consecutive forward updates.")
        candidate = candidate.detach().to(self.features)
        if candidate.shape != self.features.shape:
            raise ValueError("Memory feature shape changed within the sequence.")
        gate = gate.view(-1, 1, 1, 1).to(self.features)
        self.features = (1.0 - gate) * self.features + gate * candidate
        self.last_frame_idx = frame_idx
        self.updates += 1


class LearnedStateVideoPredictor(SAM2VideoPredictor):
    """Predict masks with one state and a selected simple update rule."""

    def __init__(
        self,
        reliability_gate_hidden_dim=32,
        state_update_mode="learned",
        fixed_ema_alpha=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if state_update_mode not in {"learned", "current", "fixed_ema"}:
            raise ValueError(f"Unknown state update mode: {state_update_mode}")
        if state_update_mode == "fixed_ema" and (
            fixed_ema_alpha is None or not 0.0 <= fixed_ema_alpha <= 1.0
        ):
            raise ValueError("fixed_ema_alpha must be in [0, 1] for fixed_ema.")
        self.reliability_gate = LearnedReliabilityGate(reliability_gate_hidden_dim)
        self.state_update_mode = state_update_mode
        self.fixed_ema_alpha = fixed_ema_alpha
        self.allow_missing_reliability_gate = state_update_mode != "learned"

    @staticmethod
    def _state_similarity(state, candidate):
        state_vector = state.float().flatten(1)
        candidate_vector = candidate.float().flatten(1)
        return F.cosine_similarity(state_vector, candidate_vector, dim=1).clamp(
            -1.0, 1.0
        )

    def propagate_in_video_preflight(self, inference_state):
        if not inference_state.get("learned_state_enabled", False):
            raise ValueError("Learned-state inference must be enabled before propagation.")
        if self._get_obj_num(inference_state) != 1:
            raise ValueError("Learned-state inference supports exactly one object.")
        super().propagate_in_video_preflight(inference_state)
        frame_idx = inference_state["learned_state_anchor_frame_idx"]
        if set(inference_state["output_dict"]["cond_frame_outputs"]) != {frame_idx}:
            raise ValueError("Learned-state inference supports one initial prompt frame.")
        output_dict = inference_state["output_dict"]
        if "learned_state" not in output_dict:
            output = output_dict["cond_frame_outputs"].get(frame_idx)
            if output is None:
                raise RuntimeError("VOS preflight did not encode the prompted frame.")
            output_dict["learned_state"] = LearnedState(frame_idx, output)

    def _reset_tracking_results(self, inference_state):
        super()._reset_tracking_results(inference_state)
        inference_state["output_dict"].pop("learned_state", None)

    def _run_single_frame_inference(self, *args, **kwargs):
        output_dict = kwargs["output_dict"]
        frame_idx = kwargs["frame_idx"]
        state = output_dict.get("learned_state")
        if state is None or kwargs["is_init_cond_frame"]:
            return super()._run_single_frame_inference(*args, **kwargs)
        if kwargs["reverse"]:
            raise ValueError("Learned-state inference supports forward propagation only.")

        current_out, pred_masks = super()._run_single_frame_inference(*args, **kwargs)
        candidate = current_out["maskmem_features"]
        if candidate is None:
            raise RuntimeError("State-update inference requires memory features.")
        predicted_iou = current_out["iou_predictions"].max(dim=-1).values
        object_score = current_out["object_score_logits"].sigmoid().flatten(1).mean(dim=1)
        similarity = self._state_similarity(state.features, candidate.to(state.features))
        if self.state_update_mode == "learned":
            quality = self.reliability_gate(predicted_iou, similarity).sigmoid()
            gate = quality * object_score
        elif self.state_update_mode == "current":
            quality = torch.ones_like(predicted_iou)
            gate = torch.ones_like(predicted_iou)
        else:
            quality = torch.full_like(predicted_iou, self.fixed_ema_alpha)
            gate = torch.full_like(predicted_iou, self.fixed_ema_alpha)
        state.update(frame_idx, candidate, gate)
        current_out["learned_state_trace"] = {
            "state_update_mode": self.state_update_mode,
            "predicted_iou": float(predicted_iou.mean().item()),
            "object_probability": float(object_score.mean().item()),
            "state_similarity": float(similarity.mean().item()),
            "quality": float(quality.mean().item()),
            "update_weight": float(gate.mean().item()),
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
            raise ValueError("Learned-state inference supports forward propagation only.")
        state = output_dict["learned_state"]
        if frame_idx != state.last_frame_idx + 1:
            raise ValueError("Learned state was read out of order.")
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
