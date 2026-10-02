"""Train RGM with the same decoder and memory-encoding path as inference."""

from __future__ import annotations

import torch

from MedSAM2.sam2.modeling.sam2_utils import get_1d_sine_pe
from MedSAM2.sam2.sam2_video_trainer import SAM2VideoTrainer
from modeling.rgm_memory import ReliabilityGatedFusion


class ReliabilityGatedMemoryTrainer(SAM2VideoTrainer):
    def __init__(self, *args, fixed_gate: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_fusion = ReliabilityGatedFusion(
            self.model.mem_dim,
            self.model.hidden_dim,
            fixed_gate=fixed_gate,
        ).to(self.device)

    def init_state(self):
        super().init_state()
        self.dynamic_state = None
        self.dynamic_state_pos = None
        self.frame_trace = []
        self._last_prediction = None

    def preprocess_frame_features(self, frame_features, batch_size, num_frames):
        prepared = []
        for frame_feature in self.unbind_frame_features(frame_features, num_frames):
            feature_maps = frame_feature["backbone_fpn"][-self.num_feature_levels :]
            vision_feats = [feature.flatten(2).permute(2, 0, 1) for feature in feature_maps]
            features = [
                feature.permute(1, 2, 0).reshape(batch_size, -1, *feature_map.shape[-2:])
                for feature, feature_map in zip(vision_feats, feature_maps)
            ]
            prepared.append({
                "image_embed": features[-1],
                "high_res_feats": features[:-1],
                "backbone_fpn": feature_maps,
                "vision_pos_enc": frame_feature["vision_pos_enc"][-self.num_feature_levels :],
            })
        return prepared

    @staticmethod
    def _as_inference_memory(features: torch.Tensor) -> torch.Tensor:
        return features.to(torch.bfloat16).to(torch.float32)

    def _native_memory_encoding(self, features, high_res_masks, object_score_logits, is_mask_from_pts):
        vision_feats = [feature.flatten(2).permute(2, 0, 1) for feature in features["backbone_fpn"]]
        feat_sizes = [feature.shape[-2:] for feature in features["backbone_fpn"]]
        memory_features, memory_positions = self.model._encode_new_memory(
            vision_feats, feat_sizes, high_res_masks, object_score_logits, is_mask_from_pts
        )
        return self._as_inference_memory(memory_features), memory_positions[-1]

    def _record_state(self, frame_idx, candidate, gate, previous_state=None):
        prediction = self._last_prediction
        if prediction is None:
            raise RuntimeError("RGM state was updated without a decoder prediction.")
        self.frame_trace.append({
            "frame_idx": frame_idx,
            "mask_logits": prediction["mask_logits"].detach().cpu(),
            "predicted_iou": prediction["predicted_iou"].detach().cpu(),
            "object_pointer": prediction["object_pointer"].detach().cpu(),
            "candidate": candidate.detach().cpu(),
            "previous_state": None if previous_state is None else previous_state.detach().cpu(),
            "state": self.dynamic_state.detach().cpu(),
            "gate": None if gate is None else gate.detach().cpu(),
        })

    def _initialize_memory(self, features, masks, object_score_logits):
        del masks, object_score_logits
        prediction = self._last_prediction
        if prediction is None:
            raise RuntimeError("RGM memory needs the prompt-frame decoder prediction.")
        candidate, position = self._native_memory_encoding(
            features, prediction["high_res_masks"], prediction["object_score_logits"], True
        )
        self.dynamic_state = candidate
        self.dynamic_state_pos = position
        self.maskmem_features = None
        self.maskmem_pos_enc = None
        self._record_state(0, candidate, gate=None)
        return self.dynamic_state, self.dynamic_state_pos

    def _update_memory(self, features, masks=None, memory=None, object_score_logits=None):
        del masks, memory, object_score_logits
        if self.dynamic_state is None:
            raise RuntimeError("RGM state was not initialized.")
        prediction = self._last_prediction
        if prediction is None:
            raise RuntimeError("RGM update needs the current decoder prediction.")
        candidate, _ = self._native_memory_encoding(
            features, prediction["high_res_masks"], prediction["object_score_logits"], False
        )
        previous_state = self.dynamic_state
        self.dynamic_state = self.state_fusion(
            previous_state,
            candidate,
            features["backbone_fpn"][-1],
        )
        self._record_state(
            self.current_frame_idx, candidate, self.state_fusion.last_gate, previous_state
        )
        return self.dynamic_state, self.dynamic_state_pos

    def _prepare_memory(self):
        if self.dynamic_state is None or self.dynamic_state_pos is None:
            raise RuntimeError("RGM state was not initialized.")
        memory_chunks = [self.dynamic_state.flatten(2).permute(2, 0, 1)]
        position_chunks = [
            (self.dynamic_state_pos + self.model.maskmem_tpos_enc[0].view(1, -1, 1, 1))
            .flatten(2)
            .permute(2, 0, 1)
        ]
        num_obj_ptr_tokens = self._append_native_pointer_history(memory_chunks, position_chunks)
        return torch.cat(memory_chunks), torch.cat(position_chunks), num_obj_ptr_tokens

    def _append_native_pointer_history(self, memory_chunks, position_chunks) -> int:
        if not self.model.use_obj_ptrs_in_encoder:
            return 0
        max_pointers = min(self.video_num_frames, self.model.max_obj_ptrs_in_encoder)
        positions_and_pointers = [(self.current_frame_idx, self.obj_ptrs[0])]
        for distance in range(1, max_pointers):
            pointer_index = self.current_frame_idx - distance
            if pointer_index <= 0:
                break
            positions_and_pointers.append((distance, self.obj_ptrs[pointer_index]))
        positions, pointers = zip(*positions_and_pointers)
        pointers = torch.stack(pointers)
        if self.model.add_tpos_enc_to_obj_ptrs:
            position_dim = (
                self.model.hidden_dim
                if self.model.proj_tpos_enc_in_obj_ptrs
                else self.model.mem_dim
            )
            temporal_positions = get_1d_sine_pe(
                torch.tensor(positions, device=pointers.device) / (max_pointers - 1),
                dim=position_dim,
            )
            pointer_positions = self.model.obj_ptr_tpos_proj(temporal_positions)
            pointer_positions = pointer_positions.unsqueeze(1).expand(
                -1, self.batch_size, self.model.mem_dim
            )
        else:
            pointer_positions = pointers.new_zeros(
                len(positions), self.batch_size, self.model.mem_dim
            )
        if self.model.mem_dim < self.model.hidden_dim:
            pointer_width = self.model.hidden_dim // self.model.mem_dim
            pointers = pointers.reshape(-1, self.batch_size, pointer_width, self.model.mem_dim)
            pointers = pointers.permute(0, 2, 1, 3).flatten(0, 1)
            pointer_positions = pointer_positions.repeat_interleave(pointer_width, dim=0)
        memory_chunks.append(pointers)
        position_chunks.append(pointer_positions)
        return pointers.shape[0]

    def _point_inputs_from_boxes(self, bboxes):
        return {
            "point_coords": bboxes.reshape(self.batch_size, 2, 2),
            "point_labels": torch.tensor(
                [[2, 3]], dtype=torch.int32, device=bboxes.device
            ).expand(self.batch_size, -1),
        }

    def _decode(self, features, point_inputs, is_init_cond_frame):
        current_feats = [feature.flatten(2).permute(2, 0, 1) for feature in features["backbone_fpn"]]
        current_pos = [feature.flatten(2).permute(2, 0, 1) for feature in features["vision_pos_enc"]]
        feat_sizes = [feature.shape[-2:] for feature in features["backbone_fpn"]]
        if is_init_cond_frame:
            pixels_with_memory = self.model._prepare_memory_conditioned_features(
                frame_idx=0,
                is_init_cond_frame=True,
                current_vision_feats=current_feats[-1:],
                current_vision_pos_embeds=current_pos[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict={},
                num_frames=self.num_frames,
            )
        else:
            memory, memory_pos, num_obj_ptr_tokens = self._prepare_memory()
            fused = self.model.memory_attention(
                curr=current_feats[-1:],
                curr_pos=current_pos[-1:],
                memory=memory,
                memory_pos=memory_pos,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            )
            pixels_with_memory = fused.permute(1, 2, 0).reshape_as(features["backbone_fpn"][-1])
        (
            _,
            _,
            iou_predictions,
            low_res_masks,
            high_res_masks,
            object_pointer,
            object_score_logits,
        ) = self.model._forward_sam_heads(
            backbone_features=pixels_with_memory,
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=features["high_res_feats"],
            multimask_output=self.model._use_multimask(is_init_cond_frame, point_inputs),
        )
        predicted_mask, predicted_logits = self._postprocess_masks(low_res_masks)
        predicted_iou = iou_predictions.max(dim=-1, keepdim=True).values
        self.obj_ptrs.append(object_pointer)
        self._last_prediction = {
            "mask_logits": predicted_logits,
            "high_res_masks": high_res_masks,
            "predicted_iou": predicted_iou,
            "object_pointer": object_pointer,
            "object_score_logits": object_score_logits,
        }
        return predicted_mask, predicted_logits, predicted_iou, object_score_logits

    def forward(self, videos, bboxes, labels=None, video_num_frames=None):
        if labels is not None:
            raise ValueError("RGM updates use predicted masks; pass labels=None.")
        self.init_state()
        batch_size, num_frames, channels, height, width = videos.shape
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.video_num_frames = num_frames if video_num_frames is None else int(video_num_frames)
        if self.video_num_frames < num_frames:
            raise ValueError("video_num_frames cannot be smaller than the clip length.")
        self._orig_hw = [height, width]
        all_features = self.model.forward_image(
            videos.reshape(batch_size * num_frames, channels, height, width)
        )
        frame_features = {
            key: value.reshape(batch_size, num_frames, *value.shape[1:])
            if not isinstance(value, list)
            else [item.reshape(batch_size, num_frames, *item.shape[1:]) for item in value]
            for key, value in all_features.items()
        }
        frames = self.preprocess_frame_features(frame_features, batch_size, num_frames)
        first_masks, first_logits, first_ious, first_scores = self._decode(
            frames[0], self._point_inputs_from_boxes(bboxes), True
        )
        self._initialize_memory(frames[0], first_masks, first_scores)
        all_masks, all_logits, all_ious = [first_masks], [first_logits], [first_ious]
        for frame_idx in range(1, num_frames):
            self.current_frame_idx = frame_idx
            masks, logits, ious, scores = self._decode(frames[frame_idx], None, False)
            all_masks.append(masks)
            all_logits.append(logits)
            all_ious.append(ious)
            self._update_memory(frames[frame_idx], masks, object_score_logits=scores)
        return all_masks, all_logits, all_ious



def freeze_except_rgm(trainer: ReliabilityGatedMemoryTrainer):
    for parameter in trainer.model.parameters():
        parameter.requires_grad = False
    for parameter in trainer.state_fusion.gate_projector.parameters():
        parameter.requires_grad = trainer.state_fusion.fixed_gate is None
    for parameter in trainer.state_fusion.rde_fusion.parameters():
        parameter.requires_grad = True


def rgm_optimizer(
    trainer: ReliabilityGatedMemoryTrainer,
    gate_lr: float = 1e-4,
    fusion_lr: float = 1e-5,
    weight_decay: float = 1e-4,
):
    freeze_except_rgm(trainer)
    groups = []
    if trainer.state_fusion.fixed_gate is None:
        groups.append({"params": trainer.state_fusion.gate_projector.parameters(), "lr": gate_lr})
    groups.append({"params": trainer.state_fusion.rde_fusion.parameters(), "lr": fusion_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
