"""Experimental, exact-checked MPS spatial-position RoPE adapter for Lean.

Position lookup is performed with MPS tensor operations. It reconstructs the
original grid index and temporal offset for each spatial key, and rejects any
position that cannot be reconstructed exactly. This is a new intervention,
not the Lean author baseline or a speed-validated production implementation.
"""
from contextlib import contextmanager
import hashlib
from types import MethodType

import torch


def decode_mps_positions(memory_pos, pointer_count, grid, offsets, chunk=128):
    if memory_pos.device.type != 'mps' or grid.device.type != 'mps' or offsets.device.type != 'mps':
        raise ValueError('Position lookup requires MPS tensors')
    if memory_pos.dtype != torch.float32 or grid.dtype != torch.float32 or offsets.dtype != torch.float32:
        raise ValueError('Position lookup requires FP32')
    if memory_pos.ndim != 3 or memory_pos.shape[1] != 1 or grid.ndim != 2 or grid.shape[0] != 4096 or grid.shape[1] % 2 or offsets.ndim != 2 or offsets.shape[1] != grid.shape[1]:
        raise ValueError('Unexpected position geometry')
    if not 0 <= pointer_count <= memory_pos.shape[0] or chunk < 1:
        raise ValueError('Invalid position lookup arguments')
    # PositionEmbeddingSine concatenates separable y and x halves.
    channels=grid.shape[1]//2
    ygrid=grid[::64,:channels]
    xgrid=grid[:64,channels:]
    if not torch.equal(grid[:,:channels].reshape(64,64,channels),ygrid[:,None,:].expand(64,64,channels)):
        raise ValueError('Position grid y is not separable')
    if not torch.equal(grid[:,channels:].reshape(64,64,channels),xgrid[None,:,:].expand(64,64,channels)):
        raise ValueError('Position grid x is not separable')
    diagonal=torch.eye(64,device=grid.device,dtype=torch.bool)
    for axis in (ygrid,xgrid):
        if bool((((axis[:,None,:]==axis[None,:,:]).all(-1)) & ~diagonal).any()):
            raise ValueError('Spatial axis positions are ambiguous')
    offsets=torch.cat((offsets,torch.zeros((1,grid.shape[1]),device=grid.device,dtype=grid.dtype)))
    yc=ygrid[None,:,:]+offsets[:,None,:channels]
    xc=xgrid[None,:,:]+offsets[:,None,channels:]
    vectors=memory_pos[:memory_pos.shape[0]-pointer_count,0,:]
    indices=[]
    for start in range(0,len(vectors),chunk):
        x=vectors[start:start+chunk]
        # Compare exact sums of squared differences in each spatial axis.
        yd=((x[:,None,None,:channels]-yc[None,:,:,:])**2).sum(-1)
        xd=((x[:,None,None,channels:]-xc[None,:,:,:])**2).sum(-1)
        ycost,yidx=yd.min(-1); xcost,xidx=xd.min(-1)
        ts=(ycost+xcost).argmin(-1)
        row=torch.arange(len(x),device=x.device)
        spatial=yidx[row,ts]*64+xidx[row,ts]
        if not torch.equal(grid[spatial]+offsets[ts],x):
            raise ValueError('Memory positions failed exact spatial reconstruction')
        same_cost=(ycost+xcost)==0
        if bool((same_cost & ((yidx*64+xidx)!=spatial[:,None])).any()):
            raise ValueError('Memory position maps to multiple spatial indices')
        indices.append(spatial)
    return torch.cat(indices) if indices else torch.empty(0,dtype=torch.long,device=grid.device)


@contextmanager
def indexed_spatial_rope_mps(model, observations):
    from sam2.modeling import memory_attention as module
    if model.device.type!='mps' or any(layer.toma_drop_ratio_ca_kv != 0 for layer in model.memory_attention.layers):
        raise ValueError('MPS indexed RoPE requires MPS and unmerged attention keys')
    with torch.no_grad():
        grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64,device=model.device)).flatten(2).permute(2,0,1)[:,0,:].contiguous()
        offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).to(model.device)
        freq=[layer.cross_attn_image.compute_cis(end_x=64,end_y=64).to(model.device) for layer in model.memory_attention.layers]
        if any(not torch.equal(x,freq[0]) for x in freq[1:]):raise ValueError('Attention rotary grids differ')
        full_freq=freq[0]
    attention=model.memory_attention
    original_forward=attention.forward
    def forward(self,*args,**kwargs):
        curr=kwargs['curr'];curr=curr[0] if isinstance(curr,list) else curr
        ptr=int(kwargs['num_obj_ptr_tokens'])
        if curr.shape[0]!=4096 or curr.shape[1]!=1:raise ValueError('Unexpected query grid')
        indices=decode_mps_positions(kwargs['memory_pos'],ptr,grid,offsets)
        if not len(indices):raise ValueError('No spatial memory positions')
        old_query,old_key=module._apply_rotary_repeat_2d,module._repeat_freqs_for_k
        calls={'query':0,'key':0}
        def rotate_query(x,_base,_length):
            if x.shape[-2]!=4096:raise ValueError('Unexpected query length')
            calls['query']+=1
            return old_query(x,full_freq,4096)
        def key_frequencies(base,repeats):
            if base.shape[0]*repeats!=len(indices) or base.shape[1]!=full_freq.shape[1]:raise ValueError('Unexpected key length')
            calls['key']+=1
            return full_freq[indices]
        module._apply_rotary_repeat_2d=rotate_query;module._repeat_freqs_for_k=key_frequencies
        try:output=original_forward(*args,**kwargs)
        finally:module._apply_rotary_repeat_2d=old_query;module._repeat_freqs_for_k=old_key
        if calls!={'query':len(self.layers),'key':len(self.layers)}:raise ValueError('Unexpected rotary calls')
        # The CPU copy is only for the engineering receipt; decoding and rotary
        # operations themselves remain on MPS and never fall back to CPU.
        indices_sha256=hashlib.sha256(indices.detach().cpu().numpy().tobytes()).hexdigest()
        observations.append({'spatial_keys':len(indices),'pointer_keys':ptr,'query_rotations':calls['query'],'key_rotations':calls['key'],'indices_sha256':indices_sha256})
        return output
    attention.forward=MethodType(forward,attention)
    try:yield
    finally:attention.forward=original_forward
