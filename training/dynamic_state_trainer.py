"""Archived gradient-enabled trainer for the compact-memory ablation."""

import math

import torch

from MedSAM2.sam2.sam2_video_trainer import SAM2VideoTrainer
from modeling.fusion import AdaptiveStateFusion


class DynamicStateVideoTrainer(SAM2VideoTrainer):
    """Replace SAM2VideoTrainer's spatial-memory list with one fused state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_fusion = AdaptiveStateFusion(self.model.mem_dim).to(self.device)
        high_resolution = self.model.image_size // 4
        self._bb_feat_sizes = [(high_resolution // (2**level),) * 2 for level in range(3)]

    def preprocess_frame_features(self, frame_features, batch_size, num_frames):
        """MPS-safe copy of the upstream method: reshape non-contiguous features."""
        prepared = []
        for frame_idx, frame_feature in enumerate(self.unbind_frame_features(frame_features, num_frames)):
            feature_maps = frame_feature["backbone_fpn"][-self.num_feature_levels :]
            vision_feats = [feature.flatten(2).permute(2, 0, 1) for feature in feature_maps]
            if frame_idx == 0 and self.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            feats = [
                feature.permute(1, 2, 0).reshape(batch_size, -1, *size)
                for feature, size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
            ][::-1]
            prepared.append({
                "image_embed": feats[-1], "high_res_feats": feats[:-1],
                "backbone_fpn": frame_feature["backbone_fpn"][-self.num_feature_levels :],
                "vision_pos_enc": frame_feature["vision_pos_enc"][-self.num_feature_levels :],
            })
        return prepared

    def _initialize_memory(self, features, masks, object_score_logits):
        encoded = self._extract_memory_features(features, masks, object_score_logits)
        self.dynamic_state = encoded["vision_features"]
        self.dynamic_state_pos = encoded["vision_pos_enc"]
        self.maskmem_features = None
        self.maskmem_pos_enc = None
        return self.dynamic_state, self.dynamic_state_pos

    def _update_memory(self, features, masks, memory=None, object_score_logits=None):
        encoded = self._extract_memory_features(features, masks, object_score_logits)
        candidate = encoded["vision_features"]
        tokens, batch_size, channels = candidate.shape
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError("Adaptive fusion requires a square memory-token grid.")
        previous_grid = self.dynamic_state.permute(1, 2, 0).reshape(batch_size, channels, side, side)
        candidate_grid = candidate.permute(1, 2, 0).reshape(batch_size, channels, side, side)
        probability = torch.nn.functional.interpolate(
            masks, size=(side, side), mode="bilinear", align_corners=False
        )
        fused = self.state_fusion(previous_grid, candidate_grid, probability)
        self.dynamic_state = fused.flatten(2).permute(2, 0, 1)
        self.dynamic_state_pos = encoded["vision_pos_enc"]
        return self.dynamic_state, self.dynamic_state_pos

    def _prepare_memory(self, memory):
        # The lists below are temporary inputs to the upstream pointer-preparation
        # code; no spatial history is retained between frames.
        return super()._prepare_memory(([self.dynamic_state], [self.dynamic_state_pos]))
