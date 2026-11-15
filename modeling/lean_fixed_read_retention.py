"""Retain ranked historical source tokens on one immutable attention read."""
import numpy as np
import torch
from modeling.cross_frame_token_pool import TokenGroup
from modeling.lean_region_read_intervention import remove_ranked_tokens


def retain_ranked_tokens(memory, positions, pointer_count, segments, protected, method, count, seed=0):
    """Keep exactly ``count`` ranked eligible tokens and all noneligible tokens.

    Reuse the existing selector and its validation. Its returned deletion indices
    are the high-ranked indices; here they identify the tokens to retain.
    """
    chosen_segments=segments
    if method=='shuffled_contrast':
        rng=np.random.default_rng(seed);chosen_segments=[]
        for item in segments:
            g=item['group'];permutation=rng.permutation(len(g.token_ids))
            shuffled=TokenGroup(g.frame,g.token_ids,g.usage_rank,np.asarray(g.contrast)[permutation])
            chosen_segments.append({**item,'group':shuffled})
        method='contrast'
    _,_,chosen=remove_ranked_tokens(memory,positions,pointer_count,chosen_segments,protected,method,count,seed=seed)
    eligible=[]
    for item in segments:
        start=item['start'];eligible.extend(range(start,start+len(item['group'].token_ids)))
    chosen_set=set(chosen)
    if len(eligible)!=len(set(eligible)) or not chosen_set<=set(eligible):raise ValueError('Eligible source spans are invalid')
    removed=sorted(set(eligible)-chosen_set)
    keep=torch.ones(memory.shape[0],dtype=torch.bool,device=memory.device)
    if removed:keep[torch.tensor(removed,device=memory.device)]=False
    out=memory[keep];out_positions=positions[keep.to(positions.device)]
    if pointer_count and (not torch.equal(out[-pointer_count:],memory[-pointer_count:]) or not torch.equal(out_positions[-pointer_count:],positions[-pointer_count:])):
        raise ValueError('Pointer tail changed')
    return out,out_positions,chosen,removed
