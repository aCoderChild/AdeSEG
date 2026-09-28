"""Frozen-MedSAM2 trainer for reliability-gated recurrent memory experiments."""

from __future__ import annotations

import math

import torch

from MedSAM2.sam2.modeling.sam2_utils import get_1d_sine_pe
from MedSAM2.sam2.sam2_video_trainer import SAM2VideoTrainer
from modeling.rgm_memory import ReliabilityGatedFusion


class ReliabilityGatedMemoryTrainer(SAM2VideoTrainer):
    """Use one spatial state while retaining MedSAM2's native pointer policy.

    The state update always receives predicted masks because callers must pass
    ``labels=None``.  Labels are used only by the external segmentation loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_fusion = ReliabilityGatedFusion(self.model.mem_dim).to(self.device)
        high_resolution = self.model.image_size // 4
        self._bb_feat_sizes = [(high_resolution // (2**level),) * 2 for level in range(3)]

    def init_state(self):
        super().init_state()
        self.dynamic_state = None
        self.dynamic_state_pos = None

    def preprocess_frame_features(self, frame_features, batch_size, num_frames):
        """Reshape features without relying on view-contiguity on MPS."""
        prepared = []
        for frame_idx, frame_feature in enumerate(self.unbind_frame_features(frame_features, num_frames)):
            feature_maps = frame_feature["backbone_fpn"][-self.num_feature_levels :]
            vision_feats = [feature.flatten(2).permute(2, 0, 1) for feature in feature_maps]
            if frame_idx == 0 and self.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            features = [
                feature.permute(1, 2, 0).reshape(batch_size, -1, *size)
                for feature, size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
            ][::-1]
            prepared.append({
                "image_embed": features[-1],
                "high_res_feats": features[:-1],
                "backbone_fpn": frame_feature["backbone_fpn"][-self.num_feature_levels :],
                "vision_pos_enc": frame_feature["vision_pos_enc"][-self.num_feature_levels :],
            })
        return prepared

    def _initialize_memory(self, features, masks, object_score_logits):
        encoded = self._extract_memory_features(features, masks, object_score_logits)
        self.dynamic_state = encoded["vision_features"]
        self.dynamic_state_pos = encoded["vision_pos_enc"]
        # Explicitly remove the upstream spatial-memory list from this trainer.
        self.maskmem_features = None
        self.maskmem_pos_enc = None
        return self.dynamic_state, self.dynamic_state_pos

    def _update_memory(self, features, masks, memory=None, object_score_logits=None):
        del memory
        if self.dynamic_state is None:
            raise RuntimeError("RGM state was not initialized.")
        encoded = self._extract_memory_features(features, masks, object_score_logits)
        candidate = encoded["vision_features"]
        tokens, batch_size, channels = candidate.shape
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError("RGM requires a square mask-memory feature grid.")
        previous_grid = self.dynamic_state.permute(1, 2, 0).reshape(batch_size, channels, side, side)
        candidate_grid = candidate.permute(1, 2, 0).reshape(batch_size, channels, side, side)
        if self._last_predicted_iou is None:
            raise RuntimeError("RGM update needs the current decoder-predicted IoU.")
        fused = self.state_fusion(previous_grid, candidate_grid, self._last_predicted_iou)
        self.dynamic_state = fused.flatten(2).permute(2, 0, 1)
        # Memory positional encoding is fixed for a spatial grid in MedSAM2.  The
        # anchor encoding therefore remains the state encoding used at inference.
        return self.dynamic_state, self.dynamic_state_pos

    def _prepare_memory(self, memory):
        del memory
        if self.dynamic_state is None or self.dynamic_state_pos is None:
            raise RuntimeError("RGM state was not initialized.")
        memory_chunks = [self.dynamic_state]
        position_chunks = [
            self.dynamic_state_pos + self.model.maskmem_tpos_enc[0].view(1, 1, -1)
        ]
        num_obj_ptr_tokens = self._append_native_pointer_history(memory_chunks, position_chunks)
        return (
            torch.cat(memory_chunks, dim=0),
            torch.cat(position_chunks, dim=0),
            num_obj_ptr_tokens,
        )

    def _append_native_pointer_history(self, memory_chunks, position_chunks) -> int:
        """Match the native prompt-anchor plus recent-pointer selection policy."""
        if not self.model.use_obj_ptrs_in_encoder:
            return 0
        max_pointers = min(self.num_frames, self.model.max_obj_ptrs_in_encoder)
        positions_and_pointers = [(self.current_frame_idx, self.obj_ptrs[0])]
        for distance in range(1, max_pointers):
            pointer_index = self.current_frame_idx - distance
            if pointer_index <= 0:
                break
            positions_and_pointers.append((distance, self.obj_ptrs[pointer_index]))
        positions, pointers = zip(*positions_and_pointers)
        pointers = torch.stack(pointers, dim=0)
        if self.model.add_tpos_enc_to_obj_ptrs:
            position_dim = self.model.hidden_dim if self.model.proj_tpos_enc_in_obj_ptrs else self.model.mem_dim
            temporal_positions = get_1d_sine_pe(
                torch.tensor(positions, device=pointers.device) / (max_pointers - 1),
                dim=position_dim,
            )
            pointer_positions = self.model.obj_ptr_tpos_proj(temporal_positions)
            pointer_positions = pointer_positions.unsqueeze(1).expand(-1, self.batch_size, self.model.mem_dim)
        else:
            pointer_positions = pointers.new_zeros(len(positions), self.batch_size, self.model.mem_dim)
        if self.model.mem_dim < self.model.hidden_dim:
            pointers = pointers.reshape(
                -1, self.batch_size, self.model.hidden_dim // self.model.mem_dim, self.model.mem_dim
            )
            pointers = pointers.permute(0, 2, 1, 3).flatten(0, 1)
            pointer_positions = pointer_positions.repeat_interleave(
                self.model.hidden_dim // self.model.mem_dim, dim=0
            )
        memory_chunks.append(pointers)
        position_chunks.append(pointer_positions)
        return pointers.shape[0]

    def _predict_frame(self, features, memory, prev_mask=None):
        del prev_mask
        memory, memory_pos_embed, num_obj_ptr_tokens = self._prepare_memory(memory)
        current_vision_feats = [feature.flatten(2).permute(2, 0, 1) for feature in features["backbone_fpn"]]
        current_vision_pos = [feature.flatten(2).permute(2, 0, 1) for feature in features["vision_pos_enc"]]
        pixels_with_memory = self.model.memory_attention(
            curr=current_vision_feats[-1:],
            curr_pos=current_vision_pos[-1:],
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pixels_with_memory = pixels_with_memory.permute(1, 2, 0).reshape(
            *features["backbone_fpn"][-1].shape
        )
        (
            _,
            _,
            iou_predictions,
            low_res_masks,
            _,
            object_pointer,
            object_score_logits,
        ) = self.model._forward_sam_heads(
            backbone_features=pixels_with_memory,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=features["high_res_feats"],
            multimask_output=False,
        )
        self.obj_ptrs.append(object_pointer)
        self._last_predicted_iou = iou_predictions[:, -1]
        predicted_mask, predicted_logits = self._postprocess_masks(low_res_masks)
        return predicted_mask, predicted_logits, self._last_predicted_iou, object_score_logits

    def forward(self, videos, bboxes, labels=None):
        if labels is not None:
            raise ValueError("RGM training must use predicted masks for memory updates; pass labels=None.")
        self._last_predicted_iou = None
        return super().forward(videos, bboxes, labels=None)


def freeze_except_rgm(trainer: ReliabilityGatedMemoryTrainer):
    """Freeze MedSAM2 and leave only the recurrent fusion module trainable."""
    for parameter in trainer.model.parameters():
        parameter.requires_grad = False
    for parameter in trainer.state_fusion.parameters():
        parameter.requires_grad = True
    return trainer.state_fusion


def rgm_optimizer(trainer: ReliabilityGatedMemoryTrainer, lr=1e-4, weight_decay=1e-4):
    fusion = freeze_except_rgm(trainer)
    return torch.optim.AdamW(fusion.parameters(), lr=lr, weight_decay=weight_decay)
