"""Deterministic global selection over already-retained Lean memory tokens.

This is a selection primitive, not a predictor hook. The caller is responsible
for reserving protected prompt, newest-memory, insurance and auxiliary state.
Scores must be computed from automatic predictions without labels.
"""
from dataclasses import dataclass
import numpy as np
import torch

@dataclass(frozen=True)
class TokenGroup:
    frame: int
    token_ids: np.ndarray
    usage_rank: np.ndarray
    contrast: np.ndarray

def tied_percentile(values):
    """Fixed within-source empirical percentile with equal values tied."""
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or not a.size or not np.isfinite(a).all():raise ValueError('Invalid scores')
    if a.size==1:return np.full(1,.5,dtype=np.float64)
    order=np.argsort(a,kind='stable');sorted_values=a[order]
    boundaries=np.r_[0,np.flatnonzero(sorted_values[1:]!=sorted_values[:-1])+1,len(a)]
    result=np.empty(len(a),dtype=np.float64)
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        result[order[begin:end]]=(begin+end-1)/(2*(len(a)-1))
    return result

def _spatial_order(ids):
    """Deterministic bit-reversed order over the original 64x64 grid IDs."""
    x=np.asarray(ids,dtype=np.uint16)
    reverse=np.zeros(len(x),dtype=np.uint16)
    for bit in range(12):
        reverse|=((x>>bit)&1)<<(11-bit)
    return np.argsort(reverse,kind='stable')

def select_global_tokens(groups,budget,strategy):
    """Choose token indices per chronological source under one fixed budget.

    ``usage`` uses source-frame percentile ranks frozen before pruning;
    ``contrast`` uses fixed half-budget foreground and background rankings,
    each combining usage rank and signed cosine contrast in [-1,1];
    ``distributed`` uses deterministic round-robin by source frame. Exact
    input order is the deterministic tie-break. Previously discarded tokens
    cannot be recovered by this primitive.
    """
    if type(budget) is not int or budget<0:raise ValueError('Invalid token budget')
    if strategy not in ('usage','contrast','distributed'):raise ValueError('Unknown strategy')
    if any(not isinstance(g,TokenGroup) or type(g.frame) is not int for g in groups):raise ValueError('Invalid group')
    if any(a.frame>=b.frame for a,b in zip(groups,groups[1:])):raise ValueError('Frames must strictly increase')
    sizes=[]
    for g in groups:
        ids=np.asarray(g.token_ids);usage=np.asarray(g.usage_rank);contrast=np.asarray(g.contrast)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or usage.shape!=ids.shape or contrast.shape!=ids.shape or not np.isfinite(usage).all() or not np.isfinite(contrast).all():raise ValueError('Invalid token metadata')
        if np.unique(ids).size!=len(ids) or (ids<0).any() or (ids>=4096).any() or (usage<0).any() or (usage>1).any() or (contrast< -1).any() or (contrast>1).any():raise ValueError('Invalid token IDs or score ranges')
        sizes.append(len(ids))
    total=sum(sizes);keep=min(budget,total)
    if not keep:
        return [np.empty(0,dtype=np.int64) for _ in groups]
    if strategy=='distributed':
        chosen=[]
        orders=[_spatial_order(g.token_ids) for g in groups]
        for offset in range(max(sizes,default=0)):
            for gi,size in enumerate(sizes):
                if offset<size:chosen.append((gi,int(orders[gi][offset])))
                if len(chosen)==keep:break
            if len(chosen)==keep:break
    elif strategy=='usage':
        scored=[]
        for gi,g in enumerate(groups):
            usage=np.asarray(g.usage_rank,dtype=np.float64)
            scored.extend((float(value),gi,j) for j,value in enumerate(usage))
        # Stable Python sort preserves chronological frame order and input order
        # on equal scores, without adding a score-dependent gate or threshold.
        scored.sort(key=lambda item:-item[0])
        chosen=[(gi,j) for _,gi,j in scored[:keep]]
    else:
        foreground=[];background=[]
        for gi,g in enumerate(groups):
            usage=np.asarray(g.usage_rank,dtype=np.float64);contrast=np.asarray(g.contrast,dtype=np.float64)
            foreground.extend((float(.5*u+.25*(c+1)),gi,j) for j,(u,c) in enumerate(zip(usage,contrast)))
            background.extend((float(.5*u+.25*(1-c)),gi,j) for j,(u,c) in enumerate(zip(usage,contrast)))
        foreground.sort(key=lambda item:-item[0]);background.sort(key=lambda item:-item[0])
        chosen=[(gi,j) for _,gi,j in foreground[:(keep+1)//2]];used=set(chosen)
        for _,gi,j in background:
            if len(chosen)==keep:break
            if (gi,j) not in used:chosen.append((gi,j));used.add((gi,j))
    per_group=[[] for _ in groups]
    for gi,j in chosen:per_group[gi].append(j)
    # Preserve original memory order within each source for feature/position pairing.
    return [np.asarray(sorted(indices),dtype=np.int64) for indices in per_group]

def compact_stored_pool(entries,budget,strategy):
    """Transactionally select compressed history, keeping positions aligned.

    Each chronological entry contains author-compressed ``[K,1,C]`` feature
    and position tensors and same-order ``pool_token_ids``, ``pool_usage_rank``
    and ``pool_contrast`` tensors. The returned replacements are newly allocated;
    this function does not mutate entries, pointers, masks or input tensors.
    """
    groups=[]
    for frame,entry in entries:
        feat=entry.get('maskmem_features_seq');pos=entry.get('maskmem_pos_seq')
        ids=entry.get('pool_token_ids');usage=entry.get('pool_usage_rank');contrast=entry.get('pool_contrast')
        if any(not torch.is_tensor(x) for x in (feat,pos,ids,usage,contrast)):
            raise ValueError('Compressed memory or score metadata is missing')
        if feat.ndim!=3 or pos.ndim!=3 or feat.shape[0]!=pos.shape[0] or feat.shape[1]!=1 or pos.shape[1]!=1:
            raise ValueError('Compressed memory alignment differs')
        n=feat.shape[0]
        if any(x.ndim!=1 or x.numel()!=n for x in (ids,usage,contrast)) or ids.dtype!=torch.long:
            raise ValueError('Token metadata alignment differs')
        if not torch.isfinite(feat).all() or not torch.isfinite(pos).all():raise ValueError('Nonfinite memory')
        groups.append(TokenGroup(frame,ids.detach().cpu().numpy(),usage.detach().cpu().numpy(),contrast.detach().cpu().numpy()))
    selected=select_global_tokens(groups,budget,strategy);replacements=[]
    for (frame,entry),indices in zip(entries,selected):
        tensors={}
        for key in ('maskmem_features_seq','maskmem_pos_seq','pool_token_ids','pool_usage_rank','pool_contrast'):
            tensor=entry[key];idx=torch.as_tensor(indices,dtype=torch.long,device=tensor.device)
            tensors[key]=tensor.index_select(0,idx)
        replacements.append({'frame':frame,'indices':indices,'tensors':tensors})
    return replacements

def source_score_metadata(usage,contrast,keep_indices):
    """Freeze scores at the original 64x64 positions selected by Lean."""
    if not all(torch.is_tensor(x) for x in (usage,contrast,keep_indices)):
        raise ValueError('Expected source score and index tensors')
    u=usage.detach().float().reshape(-1);c=contrast.detach().float().reshape(-1);ids=keep_indices.detach().reshape(-1)
    if u.numel()!=4096 or c.numel()!=4096 or ids.dtype!=torch.long or not ids.numel() or not torch.isfinite(u).all() or not torch.isfinite(c).all() or (u<0).any() or (c< -1).any() or (c>1).any():
        raise ValueError('Invalid full-grid source scores')
    if (ids<0).any() or (ids>=4096).any() or torch.unique(ids).numel()!=ids.numel():raise ValueError('Invalid retained indices')
    idx=ids.to('cpu');rank=torch.from_numpy(tied_percentile(u.cpu().numpy()).astype(np.float32))
    return {'pool_token_ids':idx.clone(),'pool_usage_rank':rank[idx].clone(),'pool_contrast':c.cpu()[idx].clone()}

def apply_compacted_pool(entries,replacements):
    """Commit a prevalidated selection without deleting masks or pointers."""
    if len(entries)!=len(replacements):raise ValueError('Replacement count differs')
    for (frame,entry),item in zip(entries,replacements):
        if frame!=item['frame'] or entry.get('smr_protected',False):raise ValueError('Invalid replacement target')
        tensors=item['tensors'];n=len(item['indices'])
        if set(tensors)!={'maskmem_features_seq','maskmem_pos_seq','pool_token_ids','pool_usage_rank','pool_contrast'} or any(not torch.is_tensor(x) or x.shape[0]!=n for x in tensors.values()):
            raise ValueError('Invalid replacement tensors')
        indices=np.asarray(item['indices'])
        if indices.ndim!=1 or indices.dtype.kind not in 'iu' or (indices<0).any() or (indices>=entry['maskmem_features_seq'].shape[0]).any() or (len(indices)>1 and np.any(indices[1:]<=indices[:-1])):
            raise ValueError('Invalid replacement indices')
        for key,tensor in tensors.items():
            original=entry[key];idx=torch.as_tensor(indices,dtype=torch.long,device=original.device)
            if not torch.equal(tensor,original.index_select(0,idx)):
                raise ValueError('Replacement feature, position or metadata differs')
    for (_,entry),item in zip(entries,replacements):
        tensors=item['tensors'];n=len(item['indices'])
        if n:
            entry.update(tensors)
            entry['smr_state']='pruned';entry['smr_pruned_token_count']=n
            entry['smr_pruned_bytes']=sum(tensors[k].numel()*tensors[k].element_size() for k in ('maskmem_features_seq','maskmem_pos_seq'))
        else:
            for key in ('maskmem_features_seq','maskmem_pos_seq','pool_token_ids','pool_usage_rank','pool_contrast'):
                entry[key]=None
            entry['smr_state']='budget_evicted';entry['token_window_evicted']=True
            entry['anchor_similarity_cache']=None
