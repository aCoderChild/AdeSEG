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
    def _masked_dice(prediction_logits, target_masks):
        target_masks = F.interpolate(
            target_masks.float(),
            size=prediction_logits.shape[-2:],
            mode="nearest",
        )
        prediction = prediction_logits.sigmoid()
        numerator = 2.0 * (prediction * target_masks).flatten(1).sum(dim=1)
        denominator = prediction.flatten(1).sum(dim=1) + target_masks.flatten(1).sum(dim=1)
        return (numerator + 1.0) / (denominator + 1.0)

    @staticmethod
    def _state_similarity(state, candidate):
        state_vector = state.float().mean(dim=(-2, -1))
        candidate_vector = candidate.float().mean(dim=(-2, -1))
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
        self._add_reliability_target(current_out, kwargs, gate_logit)
        return current_out

    def _add_reliability_target(self, current_out, kwargs, gate_logit):
        gt_masks = kwargs.get("gt_masks")
        if not self.training or gt_masks is None:
            return
        current_out["state_reliability_logit"] = gate_logit
        current_out["state_reliability_target"] = self._masked_dice(
            current_out["pred_masks_high_res"], gt_masks
        ).detach()


class ReliabilityAwareMultiStepLoss(MultiStepMultiMasksAndIous):
    """Add reliability-quality supervision to the standard segmentation losses."""

    def __init__(self, weight_dict, **kwargs):
        weight_dict = dict(weight_dict)
        weight_dict.setdefault("loss_reliability", 0.0)
        super().__init__(weight_dict=weight_dict, **kwargs)

    def _forward(self, outputs, targets, num_objects):
        losses = super()._forward(outputs, targets, num_objects)
        reference = losses[CORE_LOSS_KEY]
        reliability_logit = outputs.get("state_reliability_logit")
        reliability_target = outputs.get("state_reliability_target")
        if reliability_logit is None or reliability_target is None:
            losses["loss_reliability"] = reference.new_zeros(())
        else:
            losses["loss_reliability"] = F.binary_cross_entropy_with_logits(
                reliability_logit, reliability_target
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses
