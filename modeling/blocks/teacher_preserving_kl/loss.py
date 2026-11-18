"""Standalone teacher-preservation objective block.

Re-uses the original KL computation exactly; this module owns no model
parameters and is never installed in the inference pipeline.
"""
from torch import nn
from modeling.medical_highres.teacher_loss import (
    bernoulli_teacher_kl,
    supervised_mask_teacher_loss,
)


class TeacherPreservingKLLoss(nn.Module):
    """Detached native-teacher KL loss with an explicitly configured temperature."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if not 0 < temperature <= 10:
            raise ValueError("temperature must be in (0, 10]")
        self.temperature = float(temperature)

    def forward(self, student_logits, native_teacher_logits):
        return bernoulli_teacher_kl(
            student_logits,
            native_teacher_logits,
            temperature=self.temperature,
        )
