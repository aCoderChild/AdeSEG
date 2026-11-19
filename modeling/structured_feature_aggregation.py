"""Six-stage implementation of the historical Lean-SAM2 channel-value adapter.

Architecture:
  early + deep features -> projection -> compact region context -> Q/K/V
  channel mixing -> image-space fusion -> FG/BG contrast -> mask residual.

IMPORTANT: this class inherits all registered parameters and the strict
load_trained_state() / source-hash guard from the *unaltered* legacy model.
Consequently its state_dict keys, shapes and original runtime isinstance
checks remain compatible. Saved 0.7421 author-Lean results do NOT come from
this learned adapter.
"""
import torch

from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1
from modeling.lean_implementation_digest import implementation_digest
from modeling.aggregation_blocks import (
    FeatureProjectionBlock,
    SpatialContextBlock,
    RegionAwareChannelAttentionBlock,
    ForegroundBackgroundContrastBlock,
    FeatureFusionBlock,
    ResidualMaskDecoderBlock,
)


class StructuredMultistageFeatureAggregator(MultiscaleChannelValueRefinerV1):
    """Composition of six explicit, independently testable computation blocks."""

    def load_trained_state(self, payload, *, backbone_sha256,
                           feature_pipeline_sha256, train_manifest_sha256,
                           allow_legacy_source_only=False):
        """Authenticate structured computation AND routing source before loading.

        Old source-only checkpoints can only be loaded by explicit opt-in.
        The inherited v1 state/weight/feature checks remain mandatory.
        """
        self._trained_checkpoint_loaded = False
        declared = payload.get("implementation_sha256") if isinstance(payload, dict) else None
        if declared is None and allow_legacy_source_only is True:
            pass
        elif declared != implementation_digest():
            raise ValueError(
                "Lean implementation SHA256 mismatch or missing; "
                "source-only checkpoints require explicit legacy opt-in"
            )
        return super().load_trained_state(
            payload, backbone_sha256=backbone_sha256,
            feature_pipeline_sha256=feature_pipeline_sha256,
            train_manifest_sha256=train_manifest_sha256
        )


    def forward(self, early, deep, logits):
        # Same guards as the source-bound historical implementation.
        if (not self.training or not torch.is_grad_enabled()) and not self._trained_checkpoint_loaded:
            raise RuntimeError("Untrained refinement cannot be used in evaluation")
        if (early.ndim != 4 or deep.ndim != 4 or logits.ndim != 4
                or logits.shape[1] != 1
                or early.shape[1] != self.early_channels
                or deep.shape[1] != self.deep_channels
                or early.shape[0] != deep.shape[0]
                or early.shape[0] != logits.shape[0]):
            raise ValueError("Feature or mask geometry differs")
        if (early.device != logits.device or deep.device != logits.device
                or not all(torch.isfinite(x).all() for x in (early, deep, logits))):
            raise ValueError("Inputs must be finite and on one device")

        output_size = logits.shape[-2:]

        # 1. Cross-scale alignment: early FPN and deep memory-conditioned features.
        fine_features = FeatureProjectionBlock.forward(
            early, deep, early_projection=self.early, deep_projection=self.deep,
            output_size=output_size, mask_only=(self.mode == "mask_only"))

        # 2. Compact image context and predicted foreground probability.
        context, foreground_probability = SpatialContextBlock.forward(
            fine_features, logits, attention_size=self.attention_size)

        # 3. Region-conditioned Q/K channel correlation, applied to V.
        aggregated_values = RegionAwareChannelAttentionBlock.forward(
            context, foreground_probability, mode=self.mode,
            query=self.query, key=self.key, value=self.value)

        # 4. Residual fusion at original mask resolution.
        fused_features = FeatureFusionBlock.forward(
            fine_features, aggregated_values, output_projection=self.mix_out)

        # 5. Foreground/background similarity contrast from predictions, not GT.
        contrast = ForegroundBackgroundContrastBlock.forward(
            context, foreground_probability,
            mode=self.mode, output_size=output_size)

        # 6. Residual logit decoder; zero-init head begins as an exact no-op.
        return ResidualMaskDecoderBlock.forward(
            fused_features, contrast, logits,
            mode=self.mode, decoder_head=self.head)
