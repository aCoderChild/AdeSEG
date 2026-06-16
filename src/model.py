from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .memory import DynamicMemoryState, PromptMemoryAttention, ReliabilityGatedMemoryUpdate
from .prompts import ConceptPromptEncoder, PromptBatch, build_sam_prompt_embeddings
from .reliability import AdaptiveReliabilityEstimator


@dataclass
class ReliabilityGatedOutput:
    mask_logits: torch.Tensor
    iou_scores: torch.Tensor
    reliability: torch.Tensor
    memory: DynamicMemoryState
    candidate_memory: torch.Tensor
    low_res_mask_logits: torch.Tensor


class ReliabilityGatedDynamicMemorySAM(nn.Module):
    """
    Reliability-gated dynamic memory segmentation wrapper.

    The module reuses MedSAM2/SAM2-style image encoder, prompt encoder, and mask
    decoder modules. Conceptual prompt tokens can be supplied directly or produced
    by an optional MedSAM3 text encoder through ``ConceptPromptEncoder``.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        feature_dim: int,
        image_size: int,
        concept_encoder: ConceptPromptEncoder | None = None,
        reliability_estimator: nn.Module | None = None,
        attention_heads: int = 8,
        memory_attention_strength: float = 0.0,
        use_high_res_features: bool = False,
        sam2_base_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.feature_dim = feature_dim
        self.image_size = image_size
        self.use_high_res_features = use_high_res_features
        self.sam2_base_model = sam2_base_model
        self.concept_encoder = concept_encoder
        self.prompt_memory_attention = PromptMemoryAttention(
            feature_dim,
            attention_heads,
            memory_strength=memory_attention_strength,
        )
        self.reliability_estimator = reliability_estimator or AdaptiveReliabilityEstimator(
            feature_dim
        )
        self.memory_update = ReliabilityGatedMemoryUpdate()

    @classmethod
    def from_sam2_base(
        cls,
        sam2_model: nn.Module,
        concept_encoder: ConceptPromptEncoder | None = None,
        reliability_estimator: nn.Module | None = None,
        memory_attention_strength: float = 0.0,
    ) -> "ReliabilityGatedDynamicMemorySAM":
        """Build from an instantiated MedSAM2/SAM2Base model."""

        return cls(
            image_encoder=sam2_model.image_encoder,
            prompt_encoder=sam2_model.sam_prompt_encoder,
            mask_decoder=sam2_model.sam_mask_decoder,
            feature_dim=sam2_model.hidden_dim,
            image_size=sam2_model.image_size,
            concept_encoder=concept_encoder,
            reliability_estimator=reliability_estimator,
            memory_attention_strength=memory_attention_strength,
            use_high_res_features=getattr(
                sam2_model, "use_high_res_features_in_sam", False
            ),
            sam2_base_model=sam2_model,
        )

    def forward(
        self,
        frame: torch.Tensor,
        memory: DynamicMemoryState | None = None,
        prompts: PromptBatch | None = None,
        multimask_output: bool = False,
    ) -> ReliabilityGatedOutput:
        memory = memory or DynamicMemoryState()
        frame_features, high_res_features = self.encode_frame(frame)
        sparse_prompts, dense_prompts = build_sam_prompt_embeddings(
            prompt_encoder=self.prompt_encoder,
            prompts=prompts,
            batch_size=frame.size(0),
            device=frame.device,
            concept_encoder=self.concept_encoder,
        )

        candidate_memory = self.prompt_memory_attention(
            frame_features=frame_features,
            previous_memory=memory.tensor,
            sparse_prompts=sparse_prompts,
        )

        low_res_masks, iou_scores = self.decode_mask(
            image_features=candidate_memory,
            sparse_prompts=sparse_prompts,
            dense_prompts=dense_prompts,
            multimask_output=multimask_output,
            high_res_features=high_res_features,
        )
        reliability = self.reliability_estimator(
            frame_features=frame_features,
            previous_memory=memory.tensor,
            candidate_memory=candidate_memory,
            mask_logits=low_res_masks,
        )
        decoder_confidence = iou_scores.max(dim=1, keepdim=True).values.clamp(0.0, 1.0)
        reliability = reliability * decoder_confidence
        updated_memory = self.memory_update(
            previous_memory=memory.tensor,
            candidate_memory=candidate_memory,
            reliability=reliability,
        )
        low_res_masks, iou_scores = self.decode_mask(
            image_features=updated_memory,
            sparse_prompts=sparse_prompts,
            dense_prompts=dense_prompts,
            multimask_output=multimask_output,
            high_res_features=high_res_features,
        )
        mask_logits = F.interpolate(
            low_res_masks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return ReliabilityGatedOutput(
            mask_logits=mask_logits,
            iou_scores=iou_scores,
            reliability=reliability,
            memory=DynamicMemoryState(updated_memory),
            candidate_memory=candidate_memory,
            low_res_mask_logits=low_res_masks,
        )

    def encode_image(self, frame: torch.Tensor) -> torch.Tensor:
        frame_features, _ = self.encode_frame(frame)
        return frame_features

    def encode_frame(
        self, frame: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if self.sam2_base_model is not None:
            return self._encode_frame_with_sam2_base(frame)

        encoded = self.image_encoder(frame)
        frame_features = _last_feature_map(encoded)
        high_res_features = self._high_res_features(encoded)
        return frame_features, high_res_features

    def decode_mask(
        self,
        image_features: torch.Tensor,
        sparse_prompts: torch.Tensor,
        dense_prompts: torch.Tensor,
        multimask_output: bool,
        high_res_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_features.shape[-2:] != dense_prompts.shape[-2:]:
            image_features = F.interpolate(
                image_features,
                size=dense_prompts.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        masks, ious, *_ = self.mask_decoder(
            image_embeddings=image_features,
            image_pe=self.prompt_encoder.get_dense_pe().to(image_features.device),
            sparse_prompt_embeddings=sparse_prompts,
            dense_prompt_embeddings=dense_prompts,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return masks.float(), ious

    def _high_res_features(self, encoded: Any) -> list[torch.Tensor] | None:
        if not self.use_high_res_features:
            return None
        if not isinstance(encoded, dict) or "backbone_fpn" not in encoded:
            return None

        feature_pyramid = list(encoded["backbone_fpn"])
        if len(feature_pyramid) < 3:
            return None
        if hasattr(self.mask_decoder, "conv_s0"):
            feature_pyramid[0] = self.mask_decoder.conv_s0(feature_pyramid[0])
        if hasattr(self.mask_decoder, "conv_s1"):
            feature_pyramid[1] = self.mask_decoder.conv_s1(feature_pyramid[1])
        return feature_pyramid[:2]

    def _encode_frame_with_sam2_base(
        self, frame: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        backbone_out = self.sam2_base_model.forward_image(frame)
        _, vision_feats, _, feat_sizes = self.sam2_base_model._prepare_backbone_features(
            backbone_out
        )
        if getattr(self.sam2_base_model, "directly_add_no_mem_embed", False):
            vision_feats[-1] = vision_feats[-1] + self.sam2_base_model.no_mem_embed

        batch_size = frame.size(0)
        features = [
            feat.permute(1, 2, 0).reshape(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        return features[-1], features[:-1] if self.use_high_res_features else None


def _last_feature_map(encoded: Any) -> torch.Tensor:
    """Accept common MedSAM/SAM encoder outputs and return B,C,H,W features."""

    if torch.is_tensor(encoded):
        return encoded

    if isinstance(encoded, dict):
        if "backbone_fpn" in encoded:
            return encoded["backbone_fpn"][-1]
        if "vision_features" in encoded:
            return encoded["vision_features"]
        if "image_embeddings" in encoded:
            return encoded["image_embeddings"]

    if isinstance(encoded, (list, tuple)):
        for item in reversed(encoded):
            if torch.is_tensor(item) and item.dim() == 4:
                return item

    raise TypeError("Could not find a B,C,H,W image feature map in encoder output.")
