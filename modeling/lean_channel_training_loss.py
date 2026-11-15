"""Fixed class-balanced soft BCE used only for supervised gate training."""
import torch
import torch.nn.functional as F


def balanced_soft_bce(score,target,logit_scale=8.0):
    if score.shape!=target.shape or score.ndim!=1 or not torch.isfinite(score).all() or not torch.isfinite(target).all() or (target<0).any() or (target>1).any():
        raise ValueError('Expected finite matching soft target and score vectors')
    if logit_scale!=8.0:raise ValueError('Optimization protocol fixes logit scale to 8')
    positive=target.float();negative=1-positive
    # Separate positive and negative terms: a soft occupancy contributes to
    # both classes with its respective mass; do not multiply BCE twice by y.
    logits=score.float()*logit_scale
    pos_mass=positive.sum();neg_mass=negative.sum();terms=[]
    if pos_mass>0:terms.append((F.softplus(-logits)*positive).sum()/pos_mass)
    if neg_mass>0:terms.append((F.softplus(logits)*negative).sum()/neg_mass)
    if not terms:raise ValueError('Empty target')
    return torch.stack(terms).mean()
