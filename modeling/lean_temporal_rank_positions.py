"""Local remapping of frozen nonconditioning temporal positions for diagnostics."""
import torch


def remap_temporal_positions(memory_pos, indices, grid, offsets, target_frame,
                             history, insurance_tokens, pointer_tokens, mode):
    """Rebuild each verified source block with old, ranked or permuted slots.

    `indices` are original spatial-grid indices from the indexed-RoPE decoder.
    History is ordered (source_frame, feature_count), using the previous write's
    actual selected entries. Condition, insurance and pointer positions remain
    untouched. Features, queries and other model inputs are outside this helper.
    """
    if mode not in ('noop','rank','permuted'):raise ValueError('Unknown remapping')
    if memory_pos.ndim!=3 or memory_pos.shape[1]!=1 or memory_pos.dtype!=torch.float32 or memory_pos.device.type!='cpu':raise ValueError('Expected CPU FP32 positions')
    if grid.ndim!=2 or offsets.ndim!=2 or len(offsets)!=7 or grid.shape[1]!=memory_pos.shape[2] or offsets.shape[1]!=grid.shape[1]:raise ValueError('Position geometry differs')
    if not 1<=len(history)<=6 or any(type(i) is not int or type(n) is not int or n<=0 or i>=target_frame for i,n in history) or any(x[0]>=y[0] for x,y in zip(history,history[1:])):raise ValueError('Invalid ordered history')
    if type(insurance_tokens) is not int or type(pointer_tokens) is not int or insurance_tokens<0 or pointer_tokens<0:raise ValueError('Invalid protected segments')
    end=len(memory_pos)-pointer_tokens
    if indices.ndim!=1 or len(indices)!=end or indices.dtype!=torch.int64 or (indices<0).any() or (indices>=len(grid)).any():raise ValueError('Invalid spatial indices')
    start=4096+insurance_tokens
    if start+sum(n for _,n in history)!=end:raise ValueError('History token ranges do not match memory')
    out=memory_pos.clone();events=[];cursor=start;count=len(history)
    for j,(frame,n) in enumerate(history):
        old=min(target_frame-frame,6)-1
        chosen=old if mode=='noop' else (count-1-j if mode=='rank' else (j+1 if j<count-1 else 0))
        idx=indices[cursor:cursor+n];spatial=grid[idx]
        original=(spatial+offsets[old]).unsqueeze(1)
        if not torch.equal(memory_pos[cursor:cursor+n],original):raise ValueError('Recorded source position does not match frame age and grid')
        if mode!='noop':out[cursor:cursor+n]=(spatial+offsets[chosen]).unsqueeze(1)
        events.append({'frame':frame,'tokens':n,'start':cursor,'end':cursor+n,'actual_age':target_frame-frame,'original_slot':old,'assigned_slot':chosen})
        cursor+=n
    if not torch.equal(out[:start],memory_pos[:start]) or not torch.equal(out[end:],memory_pos[end:]):raise ValueError('Protected positions changed')
    return out,events
