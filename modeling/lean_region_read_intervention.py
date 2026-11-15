"""Offline matched-token interventions on a saved, immutable memory read.

Caller supplies verified source spans; this module never infers them from labels
or changes the stored video trajectory. The model replay is a separate step.
"""
import numpy as np
import torch
from modeling.cross_frame_token_pool import TokenGroup,select_global_tokens
from modeling.lean_region_token_pool import region_indices


def locate_verified_span(memory,positions,pointer_count,source_features,source_positions,temporal_offsets,protected):
    """Locate one source only when both full feature and position sequences match.

    The temporal offsets must be the model's actual learned offsets. An
    ambiguous match is rejected rather than assigning tokens to a guessed frame.
    """
    if memory.ndim!=3 or positions.ndim!=3 or memory.shape[:2]!=positions.shape[:2] or memory.shape[1]!=1:
        raise ValueError('Unexpected read geometry')
    if type(pointer_count) is not int or not 0<=pointer_count<=len(memory):raise ValueError('Invalid pointer tail')
    if source_features.ndim!=3 or source_positions.ndim!=3 or source_features.shape[:2]!=source_positions.shape[:2] or source_features.shape[1]!=1 or len(source_features)==0:
        raise ValueError('Invalid source geometry')
    if source_features.shape[2]!=memory.shape[2] or source_positions.shape[2]!=positions.shape[2] or temporal_offsets.ndim!=2 or temporal_offsets.shape[1]!=positions.shape[2]:
        raise ValueError('Source channels differ')
    spatial=len(memory)-pointer_count;n=len(source_features);candidates=[]
    first=source_features[0].to(memory.device)
    starts=torch.nonzero((memory[:spatial]==first).all(dim=(1,2)),as_tuple=False).flatten().tolist()
    for start in starts:
        end=start+n
        if end>spatial or any(start<y and x<end for x,y in protected):continue
        if not torch.equal(memory[start:end],source_features.to(memory.device)):continue
        pos=source_positions.to(positions.device)
        if any(torch.equal(positions[start:end],pos+offset.to(positions.device)) for offset in temporal_offsets):
            candidates.append(start)
    if len(candidates)!=1:raise ValueError('Source does not have a unique verified read span')
    return candidates[0]


def remove_ranked_tokens(memory,positions,pointer_count,segments,protected,method,count,seed=0):
    """Return a matched reduced read, preserving spatial order and pointer tail.

    Each segment is {'start': int, 'group': TokenGroup, 'boundary': array}.
    Every source span must be validated by the capture caller against the
    actual feature/position tensors. Protected half-open spans include prompt,
    newest and any other memory that must never be intervened on.
    """
    if memory.ndim!=3 or positions.ndim!=3 or memory.shape[:2]!=positions.shape[:2] or memory.shape[1]!=1:
        raise ValueError('Unexpected attention tensors')
    length=memory.shape[0]
    if type(pointer_count) is not int or not 0<=pointer_count<=length or type(count) is not int or count<=0:
        raise ValueError('Invalid pointer tail or count')
    spatial=length-pointer_count
    intervals=[]
    for start,end in protected:
        if type(start) is not int or type(end) is not int or not 0<=start<end<=spatial:raise ValueError('Invalid protected span')
        intervals.append((start,end))
    groups=[];bounds=[];starts=[]
    for item in segments:
        start=item['start'];g=item['group'];n=len(g.token_ids);end=start+n
        if type(start) is not int or n==0 or not 0<=start<end<=spatial:raise ValueError('Invalid source span')
        if any(start<y and x<end for x,y in intervals):raise ValueError('Source spans overlap protected memory or another source')
        intervals.append((start,end));groups.append(g);starts.append(start);bounds.append(np.asarray(item['boundary']))
    if count>sum(len(g.token_ids) for g in groups):raise ValueError('Not enough eligible tokens')
    if method=='foreground':
        # The unbalanced foreground counterpart to the balanced contrast arm.
        ranked=[]
        select_global_tokens(groups,0,'usage')
        for i,g in enumerate(groups):
            ranked.extend((float(.5*u+.25*(c+1)),i,j) for j,(u,c) in enumerate(zip(g.usage_rank,g.contrast)))
        ranked.sort(key=lambda x:-x[0]);chosen=[[] for _ in groups]
        for _,i,j in ranked[:count]:chosen[i].append(j)
        indices=[np.asarray(sorted(x),dtype=np.int64) for x in chosen]
    elif method=='shuffled':
        rng=np.random.default_rng(seed);shuffled=[];shuffled_boundary=[]
        for g,b in zip(groups,bounds):
            permutation=rng.permutation(len(g.token_ids))
            shuffled.append(TokenGroup(g.frame,g.token_ids,g.usage_rank,np.asarray(g.contrast)[permutation]))
            shuffled_boundary.append(b[permutation])
        indices=region_indices(shuffled,shuffled_boundary,count,'three_region')
    else:
        indices=region_indices(groups,bounds,count,method)
    removed=sorted(start+int(j) for start,selected in zip(starts,indices) for j in selected)
    if len(removed)!=count or len(set(removed))!=count:raise ValueError('Intervention did not remove exact count')
    keep=torch.ones(length,dtype=torch.bool,device=memory.device);keep[torch.tensor(removed,device=memory.device)]=False
    out_memory=memory[keep];out_positions=positions[keep.to(positions.device)]
    if pointer_count and (not torch.equal(out_memory[-pointer_count:],memory[-pointer_count:]) or not torch.equal(out_positions[-pointer_count:],positions[-pointer_count:])):
        raise ValueError('Pointer tail changed')
    return out_memory,out_positions,removed
