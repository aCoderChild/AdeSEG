"""Segmentation supervision and detached native-MedSAM2 teacher preservation.

Ground-truth labels are allowed only in training; never use the training
objectives or a label-derived mask as test-time model input.
"""
import torch
from torch.nn import functional as F


def bernoulli_teacher_kl(student_logits, teacher_logits, *, temperature=1.0):
    """KL(Bernoulli teacher || Bernoulli student) with detached teacher."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 4:
        raise ValueError("Student and teacher logits must have identical BCHW geometry")
    if not 0 < temperature <= 10:
        raise ValueError("Invalid distillation temperature")
    teacher = teacher_logits.detach().float() / temperature
    student = student_logits.float() / temperature
    if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
        raise ValueError("Teacher/student logits must be finite")
    p = teacher.sigmoid()
    # BCE(student,p) - BCE(teacher,p): exactly zero for equal logits even
    # when the teacher suppresses a mask with a large negative logit.
    cross_entropy = F.binary_cross_entropy_with_logits(student, p, reduction="none")
    entropy = F.binary_cross_entropy_with_logits(teacher, p, reduction="none")
    return (cross_entropy - entropy).mean() * temperature**2


def supervised_mask_teacher_loss(student_logits, teacher_logits, target,
                                 *, teacher_weight=0.1, positive_weight=0.0):
    """BCE + soft Dice + optional native teacher KL and positive preservation."""
    if (student_logits.shape != teacher_logits.shape
            or student_logits.shape != target.shape
            or target.ndim != 4 or target.shape[1] != 1):
        raise ValueError("Logits and target must be aligned Bx1xHxW")
    if not 0 <= teacher_weight <= 10 or not 0 <= positive_weight <= 10:
        raise ValueError("Invalid loss coefficients")
    if not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError("Training targets must be finite probabilities in [0,1]")
    target = target.detach().float()
    student = student_logits.float()
    bce = F.binary_cross_entropy_with_logits(student, target)
    p = student.sigmoid()
    dice = 1 - ((2 * (p * target).sum((-2, -1)) + 1) /
                ((p + target).sum((-2, -1)) + 1)).mean()
    kl = bernoulli_teacher_kl(student, teacher_logits)
    positive = p.new_zeros(())
    if positive_weight:
        from modeling.positive_teacher_preservation_loss import positive_teacher_penalty
        positive = positive_teacher_penalty(student, teacher_logits, target)
    loss = bce + dice + teacher_weight * kl + positive_weight * positive
    return loss, {"bce": bce.detach(), "dice": dice.detach(),
                  "teacher_kl": kl.detach(), "positive_preservation": positive.detach()}
