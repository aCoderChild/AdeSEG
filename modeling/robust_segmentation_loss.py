"""Training-only robust BCE plus the existing soft-Dice objective.

No model output, object-presence decision or inference mask convention is
changed. This exploratory loss reduces the influence of very large per-pixel
cross-entropy values such as SAM2's suppressed -1024 logits.
"""
import torch
import torch.nn.functional as F


def robust_bce_dice_loss(prediction, target, *, scale=10.0):
    """Smooth ``scale * log1p(BCE/scale)`` per pixel plus soft Dice loss."""
    if prediction.shape!=target.shape or prediction.ndim!=4 or prediction.shape[1]!=1:
        raise ValueError('Prediction and target geometry must match')
    if prediction.device!=target.device or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Inputs must be finite and on one device')
    if not isinstance(scale,(int,float)) or isinstance(scale,bool) or not 0<scale<1e6:
        raise ValueError('Loss scale must be positive and finite')
    if bool((target<0).any()) or bool((target>1).any()):
        raise ValueError('Training target must be between zero and one')
    bce=F.binary_cross_entropy_with_logits(prediction,target,reduction='none')
    robust=(scale*torch.log1p(bce/scale)).mean()
    prob=prediction.sigmoid()
    dice_loss=1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
    return robust+dice_loss
