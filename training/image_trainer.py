"""Image-training building blocks for REFUGE2; orchestration remains explicit."""

from training.losses import dice_bce_loss

__all__ = ["dice_bce_loss"]
