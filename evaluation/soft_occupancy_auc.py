"""Pairwise rank AUC for soft occupancy targets, including tie handling."""
import numpy as np


def soft_occupancy_auc(scores,occupancy):
    s=np.asarray(scores,dtype=np.float64).reshape(-1);y=np.asarray(occupancy,dtype=np.float64).reshape(-1)
    if not s.size or s.shape!=y.shape or not np.isfinite(s).all() or not np.isfinite(y).all() or (y<0).any() or (y>1).any():raise ValueError('Invalid scores or occupancy')
    pos=float(y.sum());neg=float((1-y).sum())
    if pos==0 or neg==0:return None
    order=np.argsort(s,kind='stable');s=s[order];y=y[order];bounds=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1,len(s)]
    group_pos=np.add.reduceat(y,bounds[:-1]);group_neg=np.add.reduceat(1-y,bounds[:-1])
    neg_before=np.cumsum(group_neg)-group_neg
    return float(np.sum(group_pos*(neg_before+.5*group_neg))/(pos*neg))
