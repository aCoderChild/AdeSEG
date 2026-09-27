"""Image-training hooks for REFUGE2 and future adenoid frame pretraining.

The project does not duplicate the full MedSAM2 trainer here. This module keeps
only task-specific loss hooks; full MedSAM2 fine-tuning should reuse the bundled
MedSAM2 training stack with the dataset adapter/config supplied by AdeSEG.
"""

from training.losses import dice_bce_loss

__all__ = ["dice_bce_loss"]
