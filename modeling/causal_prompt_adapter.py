"""Backward-compatible name for the implicit temporal-state adapter."""

from modeling.implicit_state import ImplicitTemporalState


class CausalProbabilityStateAdapter(ImplicitTemporalState):
    """Alias retained for earlier experiment scripts."""

    def __init__(
        self,
        hidden_channels: int = 32,
        maximum_prompt_logit: float = 8.0,
        object_pointer_dim: int = 256,
        image_feature_channels: int = 256,
    ) -> None:
        super().__init__(
            hidden_channels=hidden_channels,
            image_feature_channels=image_feature_channels,
            object_pointer_dim=object_pointer_dim,
            maximum_prompt_logit=maximum_prompt_logit,
        )

    def forward_step(self, base_logits, current_object_pointer, current_image_features, state=None):
        return super().forward_step(
            mask_logits=base_logits,
            image_features=current_image_features,
            object_pointer=current_object_pointer,
            state=state,
        )
