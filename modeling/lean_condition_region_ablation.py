"""Deterministic same-read deletion ranks for automatic prompt memory regions."""
import numpy as np
import torch
import torch.nn.functional as F
from modeling.cross_frame_token_pool import _spatial_order


def prompt_region_scores(mask):
    mask=torch.as_tensor(mask,dtype=torch.float32)
    if mask.ndim!=2 or not torch.isfinite(mask).all() or (mask<0).any() or (mask>1).any():raise ValueError('Expected automatic mask in [0,1]')
    return F.interpolate(mask[None,None],size=(64,64),mode='area').reshape(-1).numpy()


def condition_ablation_indices(occupancy,usage,method,count=1024,seed=20261114):
    occ=np.asarray(occupancy,dtype=np.float64);u=np.asarray(usage,dtype=np.float64)
    if occ.shape!=(4096,) or u.shape!=(4096,) or not np.isfinite(occ).all() or not np.isfinite(u).all() or (occ<0).any() or (occ>1).any() or (u<0).any() or not isinstance(count,int) or not 0<count<=4096:raise ValueError('Invalid conditioning scores or count')
    spatial=_spatial_order(np.arange(4096));tie=np.empty(4096,dtype=np.int64);tie[spatial]=np.arange(4096)
    if method=='foreground':value=occ
    elif method=='background':value=1-occ
    elif method=='boundary':value=occ*(1-occ)
    elif method=='uniform':return np.sort(spatial[:count]).astype(np.int64)
    elif method=='shuffled':value=occ[np.random.default_rng(seed).permutation(4096)]
    elif method=='attention_oracle':value=u
    else:raise ValueError('Unknown conditioning intervention')
    return np.sort(np.lexsort((tie,-value))[:count]).astype(np.int64)


def delete_condition_tokens(memory,positions,pointer_count,indices):
    if memory.ndim!=3 or positions.ndim!=3 or memory.shape[:2]!=positions.shape[:2] or memory.shape[1]!=1 or not 0<=pointer_count<=len(memory)-4096:raise ValueError('Invalid memory layout')
    ids=np.asarray(indices)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or len(np.unique(ids))!=len(ids) or (ids<0).any() or (ids>=4096).any():raise ValueError('Invalid conditioning indices')
    keep=torch.ones(len(memory),dtype=torch.bool,device=memory.device);keep[torch.as_tensor(ids,device=memory.device)]=False
    out=memory[keep];pos=positions[keep.to(positions.device)]
    tail=len(memory)-4096
    if tail and (not torch.equal(out[-tail:],memory[4096:]) or not torch.equal(pos[-tail:],positions[4096:])):raise ValueError('Noncondition memory changed')
    return out,pos
