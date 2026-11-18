"""Stage 3: region-aware Q/K channel correlations and V aggregation."""
from modeling.multiscale_channel_value_refiner_v1 import channel_matrix


class RegionAwareChannelAttentionBlock:
    """Cross-channel value mixing; no new weights beyond the legacy projections."""

    @staticmethod
    def forward(context, foreground_probability, *, mode, query, key, value):
        attention_mode = (
            "global" if mode in ("mask_only", "global_contrast")
            else "foreground_background"
            if mode in ("foreground_background_no_contrast", "foreground_background_asinh")
            else mode
        )
        weights = channel_matrix(
            query(context), key(context), foreground_probability, attention_mode)
        values = value(context)
        return (weights @ values.flatten(2)).reshape_as(values)
