"""Trainable, reliability-gated recurrent memory for MedSAM2."""

import torch
import torch.nn.functional as F

from MedSAM2.training.loss_fns import MultiStepMultiMasksAndIous
from MedSAM2.training.model.sam2 import SAM2Train
from MedSAM2.training.trainer import CORE_LOSS_KEY
from modeling.reliability_gate import LearnedReliabilityGate


class ReliabilityGatedSAM2Train(SAM2Train):
    """SAM2Train with one recurrent spatial state and a learned update gate."""

    def __init__(
        self,
        reliability_gate_hidden_dim=32,
        freeze_base_model=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.freeze_base_model = freeze_base_model
        if freeze_base_model:
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.reliability_gate = LearnedReliabilityGate(reliability_gate_hidden_dim)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_base_model:
            for module in self.children():
                if module is not self.reliability_gate:
                    module.eval()
        return self

    @staticmethod
    def _mask_iou(prediction_logits, target_masks):
        target_masks = F.interpolate(
            target_masks.float(),
            size=prediction_logits.shape[-2:],
            mode="nearest",
        )
        prediction = prediction_logits > 0
        target = target_masks > 0
        intersection = (prediction & target).flatten(1).sum(dim=1).float()
        union = (prediction | target).flatten(1).sum(dim=1).float()
        return torch.where(union > 0, intersection / union, torch.ones_like(union))

    @staticmethod
    def _state_similarity(state, candidate):
        state_vector = state.float().flatten(1)
        candidate_vector = candidate.float().flatten(1)
        return F.cosine_similarity(state_vector, candidate_vector, dim=1).clamp(-1.0, 1.0)

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
            raise ValueError("Reliability-gated dynamic state supports forward tracking only.")
        state = output_dict["dynamic_state"]
        state_position = output_dict["dynamic_state_position"]
        dtype = current_vision_feats[-1].dtype
        memory = state.to(dtype).flatten(2).permute(2, 0, 1)
        memory_position = (
            state_position + self.maskmem_tpos_enc[0].view(1, -1, 1, 1)
        ).to(dtype).flatten(2).permute(2, 0, 1)
        fused = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_position,
            num_obj_ptr_tokens=0,
        )
        batch_size = current_vision_feats[-1].size(1)
        return fused.permute(1, 2, 0).reshape(
            batch_size, self.hidden_dim, *feat_sizes[-1]
        )

    def track_step(self, *args, **kwargs):
        current_out = super().track_step(*args, **kwargs)
        output_dict = kwargs["output_dict"]
        is_init_cond_frame = kwargs["is_init_cond_frame"]
        candidate = current_out["maskmem_features"]
        if candidate is None:
            raise RuntimeError("Reliability-gated dynamic state requires memory features.")

        if "dynamic_state" not in output_dict:
            if not is_init_cond_frame:
                raise RuntimeError("Dynamic state must be initialized from the prompt frame.")
            output_dict["dynamic_state"] = candidate.float()
            output_dict["dynamic_state_position"] = current_out["maskmem_pos_enc"][-1].float()
            return current_out

        if is_init_cond_frame:
            raise ValueError("Reliability-gated dynamic state supports one initial prompt frame.")

        state = output_dict["dynamic_state"]
        predicted_iou = current_out["multistep_pred_ious"][-1].max(dim=-1).values
        object_score = current_out["multistep_object_score_logits"][-1].sigmoid().flatten(1).mean(dim=1)
        similarity = self._state_similarity(state, candidate)
        gate_logit = self.reliability_gate(predicted_iou, object_score, similarity)
        gate = gate_logit.sigmoid().view(-1, 1, 1, 1)
        next_state = (1.0 - gate) * state + gate * candidate
        output_dict["dynamic_state"] = next_state
        self._add_quality_target(current_out, kwargs, gate_logit)
        return current_out

    def _add_quality_target(self, current_out, kwargs, gate_logit):
        gt_masks = kwargs.get("gt_masks")
        if not self.training or gt_masks is None:
            return
        current_out["state_quality_logit"] = gate_logit
        current_out["state_quality_target"] = self._mask_iou(
            current_out["pred_masks_high_res"], gt_masks
        ).detach()


class ReliabilityAwareMultiStepLoss(MultiStepMultiMasksAndIous):
    """Add actual-mask-IoU supervision to the learned update gate."""

    def __init__(self, weight_dict, **kwargs):
        weight_dict = dict(weight_dict)
        weight_dict.setdefault("loss_quality", 0.0)
        super().__init__(weight_dict=weight_dict, **kwargs)

    def _forward(self, outputs, targets, num_objects):
        losses = super()._forward(outputs, targets, num_objects)
        reference = losses[CORE_LOSS_KEY]
        quality_logit = outputs.get("state_quality_logit")
        quality_target = outputs.get("state_quality_target")
        if quality_logit is None or quality_target is None:
            losses["loss_quality"] = reference.new_zeros(())
        else:
            losses["loss_quality"] = F.smooth_l1_loss(
                quality_logit.sigmoid(), quality_target
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses
