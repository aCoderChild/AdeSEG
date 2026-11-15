"""Matched-key frozen-read exchange of spatial, newest or pointer memory."""
import torch
from modeling.lean_cross_trajectory_swap import swap_nonconditioning_spatial


def exchange_memory(recipient,donor,kind):
    """Move values and positions together while retaining recipient query/state."""
    if kind not in ('spatial','newest','pointers'):raise ValueError('Unsupported exchange')
    # Each comparison retains its recipient's own query. The two trajectories
    # can encode the same image differently; only memory comes from the donor.
    # Reuse existing strict condition, geometry and pointer-count validation.
    donor_memory_with_recipient_query=dict(donor, curr=recipient['curr'], curr_pos=recipient['curr_pos'])
    full=swap_nonconditioning_spatial(recipient,donor_memory_with_recipient_query)
    ptr=recipient['num_obj_ptr_tokens'];end=len(recipient['memory'])-ptr
    if ptr<=0 or end!=16384:raise ValueError('Expected a 16384-key spatial memory and pointer suffix')
    if kind=='spatial':return full
    start,stop=(end-4096,end) if kind=='newest' else (end,len(recipient['memory']))
    result=dict(recipient)
    for key in ('memory','memory_pos'):
        a,b=recipient[key],donor[key]
        result[key]=torch.cat((a[:start],b[start:stop],a[stop:]),dim=0)
        if not torch.equal(result[key][:start],a[:start]) or not torch.equal(result[key][stop:],a[stop:]):raise ValueError('Recipient outside-block contents changed')
    return result
