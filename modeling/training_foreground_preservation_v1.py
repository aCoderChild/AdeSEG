"""Training-only automatic-mask augmentation and positive preservation.

The dilation uses only the original model logits. Ground-truth labels enter
the preservation penalty during training only; neither function is called
by an inference model. In particular, no presence threshold is introduced.
"""
import torch
import torch.nn.functional as F


def expand_logits(logits, width):
    """Return a spatial max expansion of low-resolution automatic logits.

    A square maximum can be computed as two separable 1-D maxima. Width one
    is an exact identity, including dtype and values. Native suppressed
    constant logits remain constant; expansion does not invent an object in
    an entirely suppressed mask.
    """
    if logits.ndim != 4 or logits.shape[1] != 1 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite N x 1 x H x W logits')
    if not isinstance(width, int) or isinstance(width, bool) or width < 1 or width % 2 != 1:
        raise ValueError('Expansion width must be a positive odd integer')
    if width > min(logits.shape[-2:]):
        raise ValueError('Expansion window exceeds the mask')
    if width == 1:
        return logits
    pad = width // 2
    horizontal = F.max_pool2d(logits, (1, width), stride=1, padding=(0, pad))
    return F.max_pool2d(horizontal, (width, 1), stride=1, padding=(pad, 0))


def positive_preservation_penalty(prediction, teacher, target):
    """Penalize losing teacher confidence only on annotated training positives.

    This is a one-sided version of the project's existing positive-teacher
    idea.  An overextended teacher's negative-region pixels are unprotected;
    an increase in positive probability never incurs the penalty. Teacher
    confidence scales the penalty rather than cancelling during normalization.
    """
    if prediction.ndim != 4 or prediction.shape[1] != 1 or prediction.shape != teacher.shape or prediction.shape != target.shape:
        raise ValueError('Prediction, teacher logits and target must align')
    if prediction.device != teacher.device or prediction.device != target.device:
        raise ValueError('All tensors must be on the same device')
    if not all(torch.isfinite(x).all() for x in (prediction, teacher, target)):
        raise ValueError('All tensors must be finite')
    if bool((target < 0).any()) or bool((target > 1).any()):
        raise ValueError('Target must be in [0, 1]')
    teacher_prob = teacher.detach().float().sigmoid()
    target = target.detach().float()
    reduction = torch.relu(teacher_prob - prediction.float().sigmoid())
    return (target * teacher_prob * reduction.square()).sum() / target.sum().clamp_min(1.0)
