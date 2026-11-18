"""Training-only KL divergence from detached native MedSAM2 predictions.

The loss does not consume masks and is never part of inference routing.
"""
from torch import nn

from modeling.medical_highres.teacher_loss import bernoulli_teacher_kl


class TeacherPreservingKLLoss(nn.Module):
    """Parameter-free native-teacher KL objective, not a feature block."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if not 0 < temperature <= 10:
            raise ValueError("temperature must be in (0, 10]")
        self.temperature = float(temperature)

    def forward(self, student_logits, teacher_logits):
        return bernoulli_teacher_kl(
            student_logits, teacher_logits, temperature=self.temperature)


__all__ = ["TeacherPreservingKLLoss", "bernoulli_teacher_kl"]
