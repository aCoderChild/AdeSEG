"""Training-only preservation of confident automatic positives.

This loss uses ground truth only as a training target. It changes neither the
model's inference inputs nor its output, object-presence or mask threshold.
"""
import torch


def positive_teacher_penalty(prediction, original, target):
    """Soft GT-positive/teacher-confidence weighted probability deviation.

    Empty targets and a fully suppressed (-1024) teacher give a zero penalty;
    the separate segmentation loss is still responsible for learning them.
    """
    if prediction.ndim!=4 or prediction.shape[1]!=1 or prediction.shape!=original.shape or prediction.shape!=target.shape:
        raise ValueError('Prediction, original logits and target must align')
    if prediction.device!=original.device or prediction.device!=target.device:
        raise ValueError('All tensors must be on the same device')
    if not all(torch.isfinite(x).all() for x in (prediction,original,target)):
        raise ValueError('Inputs must be finite')
    if bool((target<0).any()) or bool((target>1).any()):
        raise ValueError('Target must be in [0, 1]')
    teacher=original.detach().sigmoid()
    weight=target.detach()*teacher
    return (weight*(prediction.sigmoid()-teacher).square()).sum()/target.detach().sum().clamp_min(1.0)
