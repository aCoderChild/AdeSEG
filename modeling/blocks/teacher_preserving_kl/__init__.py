"""Training-only native-teacher KL; never evaluated with ground-truth gates."""
from .loss import TeacherPreservingKLLoss, bernoulli_teacher_kl, supervised_mask_teacher_loss

__all__ = ["TeacherPreservingKLLoss", "bernoulli_teacher_kl", "supervised_mask_teacher_loss"]
