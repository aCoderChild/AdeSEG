"""Contribution 1: MedSAM2 high-resolution feature adapter with teacher retention."""
from .adapter import MedicalHighResolutionAdapter
from .integration import attach_medical_highres_adapter
from .checkpoints import load_reconstructed_medical_adapter
from .teacher_loss import bernoulli_teacher_kl, supervised_mask_teacher_loss

__all__ = [
    "MedicalHighResolutionAdapter",
    "attach_medical_highres_adapter",
    "load_reconstructed_medical_adapter",
    "bernoulli_teacher_kl",
    "supervised_mask_teacher_loss",
]
