"""Trainable, reliability-gated recurrent memory for MedSAM2."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from MedSAM2.training.loss_fns import MultiStepMultiMasksAndIous
from MedSAM2.training.model.sam2 import SAM2Train
from MedSAM2.training.trainer import CORE_LOSS_KEY


class ReliabilityGate(nn.Module):
    """Predict the probability that the current memory feature is trustworthy."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, predicted_iou, object_score, state_similarity, mask_entropy):
        signals = torch.stack(
            (predicted_iou, object_score, state_similarity, mask_entropy), dim=-1
        )
        return self.network(signals).squeeze(-1)


class ReliabilityGatedSAM2Train(SAM2Train):
    """SAM2Train with one recurrent spatial state and a learned update gate."""

    def __init__(
        self,
        reliability_gate_hidden_dim=32,
        mask_corruption_probability=0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 <= mask_corruption_probability <= 1.0:
            raise ValueError("mask_corruption_probability must be in [0, 1].")
        self.reliability_gate = ReliabilityGate(reliability_gate_hidden_dim)
        self.mask_corruption_probability = mask_corruption_probability

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
    def _mask_entropy(mask_logits):
        probability = mask_logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        entropy = -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())
        return entropy.flatten(1).mean(dim=1) / torch.log(torch.tensor(2.0, device=entropy.device))

    @staticmethod
    def _state_similarity(state, candidate):
        state_vector = state.float().mean(dim=(-2, -1))
        candidate_vector = candidate.float().mean(dim=(-2, -1))
        return F.cosine_similarity(state_vector, candidate_vector, dim=1).clamp(-1.0, 1.0)

    def _encode_memory_from_mask(self, current_vision_feats, feat_sizes, masks):
        batch_size = current_vision_feats[-1].size(1)
        height, width = feat_sizes[-1]
        pixel_features = current_vision_feats[-1].permute(1, 2, 0).reshape(
            batch_size, self.hidden_dim, height, width
        )
        memory_masks = masks.float()
        memory_masks = memory_masks * self.sigmoid_scale_for_mem_enc
        memory_masks = memory_masks + self.sigmoid_bias_for_mem_enc
        memory_features = self.memory_encoder(
            pixel_features, memory_masks, skip_mask_sigmoid=True
        )["vision_features"]
        if self.no_obj_embed_spatial is not None:
            object_present = (masks > 0).flatten(1).any(dim=1).float()
            memory_features = memory_features + (
                (1.0 - object_present)[..., None, None, None]
                * self.no_obj_embed_spatial[..., None, None].expand_as(memory_features)
            )
        return memory_features

    def _perturb_mask(self, masks):
        if self.mask_corruption_probability == 0.0:
            return masks
        flip = torch.rand_like(masks) < self.mask_corruption_probability
        return torch.where(flip, 1.0 - masks, masks)

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
            output_dict["dynamic_state"] = candidate
            output_dict["dynamic_state_position"] = current_out["maskmem_pos_enc"][-1]
            self._add_auxiliary_targets(current_out, candidate, kwargs)
            return current_out

        if is_init_cond_frame:
            raise ValueError("Reliability-gated dynamic state supports one initial prompt frame.")

        state = output_dict["dynamic_state"]
        predicted_iou = current_out["multistep_pred_ious"][-1].max(dim=-1).values
        object_score = current_out["multistep_object_score_logits"][-1].sigmoid().flatten(1).mean(dim=1)
        similarity = self._state_similarity(state, candidate)
        entropy = self._mask_entropy(current_out["pred_masks_high_res"])
        gate_logit = self.reliability_gate(predicted_iou, object_score, similarity, entropy)
        gate = gate_logit.sigmoid().view(-1, 1, 1, 1)
        next_state = (1.0 - gate) * state + gate * candidate
        output_dict["dynamic_state"] = next_state
        self._add_auxiliary_targets(current_out, next_state, kwargs, gate_logit)
        return current_out

    def _add_auxiliary_targets(self, current_out, state, kwargs, gate_logit=None):
        gt_masks = kwargs.get("gt_masks")
        if not self.training or gt_masks is None:
            return
        current_vision_feats = kwargs["current_vision_feats"]
        feat_sizes = kwargs["feat_sizes"]
        clean_feature = self._encode_memory_from_mask(
            current_vision_feats, feat_sizes, gt_masks
        )
        corrupted_feature = self._encode_memory_from_mask(
            current_vision_feats,
            feat_sizes,
            self._perturb_mask(gt_masks),
        )
        current_out["loss_corrupt_consistency"] = F.mse_loss(
            corrupted_feature, clean_feature.detach()
        )
        current_out["loss_state_guidance"] = F.mse_loss(state, clean_feature.detach())
        if gate_logit is not None:
            current_out["state_reliability_logit"] = gate_logit
            current_out["state_reliability_target"] = self._masked_dice(
                current_out["pred_masks_high_res"], gt_masks
            ).detach()


class ReliabilityAwareMultiStepLoss(MultiStepMultiMasksAndIous):
    """Add learned-gate, clean-state, and corrupted-mask objectives to SAM2 losses."""

    def __init__(self, weight_dict, **kwargs):
        weight_dict = dict(weight_dict)
        for key in (
            "loss_reliability",
            "loss_corrupt_consistency",
            "loss_state_guidance",
        ):
            weight_dict.setdefault(key, 0.0)
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
        for key in ("loss_corrupt_consistency", "loss_state_guidance"):
            losses[key] = outputs.get(key, reference.new_zeros(()))
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses
