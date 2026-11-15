"""Capture source spans for offline, matched-token interventions.

No alternate forward pass is executed during propagation. Each compressed
source is matched in its entirety against the actual read and temporal position;
ambiguous or absent matches fail closed. The existing capture records decoder
inputs and outputs separately.
"""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import torch
from modeling.lean_memory_capture import capture_memory_read,clone_data
from modeling.lean_region_read_intervention import locate_verified_span


@contextmanager
def capture_region_reads(model,target_frames,output,observations):
    materialize=model._materialize_memory_entry;get_feature=model._get_image_feature
    state={'frame':None,'sources':{},'captured':set()}
    offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).cpu().clone()
    def feature(self,inference_state,frame_idx,batch_size):
        if state['frame']!=int(frame_idx):state['sources']={}
        state['frame']=int(frame_idx)
        return get_feature(inference_state,frame_idx,batch_size)
    def fetch(self,prev,device,*,is_cond=False):
        result=materialize(prev,device,is_cond=is_cond)
        if state['frame'] in target_frames and not is_cond and prev is not None and torch.is_tensor(prev.get('pool_token_ids')):
            f,p=result;frame=prev.get('frame_idx');ids=prev['pool_token_ids']
            if type(frame) is not int or frame>=state['frame'] or f is None or len(f)!=len(ids):raise ValueError('Unexpected source identity')
            source={'frame':frame,'features':f,'positions':p,'token_ids':ids,
                    'usage':prev.get('pool_usage_rank'),'contrast':prev.get('pool_contrast'),'boundary':prev.get('pool_boundary')}
            if any(not torch.is_tensor(source[k]) or source[k].ndim!=1 or len(source[k])!=len(ids) for k in ('token_ids','usage','contrast','boundary')):
                raise ValueError('Source metadata unavailable')
            state['sources'][frame]=clone_data(source)
        return result
    model._get_image_feature=MethodType(feature,model);model._materialize_memory_entry=MethodType(fetch,model)
    try:
        with capture_memory_read(model,target_frames,output,observations):
            attention=model.memory_attention.forward
            def attend(self,*args,**kwargs):
                frame=state['frame'];capture=frame in target_frames and frame not in state['captured']
                if capture:
                    memory=kwargs['memory'];pos=kwargs['memory_pos'];ptr=int(kwargs['num_obj_ptr_tokens'])
                    protected=[(0,4096)];sources=[]
                    for _,source in sorted(state['sources'].items()):
                        start=locate_verified_span(memory,pos,ptr,source['features'],source['positions'],offsets,protected)
                        end=start+len(source['token_ids']);protected.append((start,end))
                        source['start']=start;sources.append(source)
                    if not sources:raise ValueError('No verified older sources')
                    folder=Path(output)/'source_snapshots';folder.mkdir(exist_ok=True)
                    torch.save({'frame':frame,'pointer_count':ptr,'sources':sources},folder/f'{frame:06d}.pt')
                    state['captured'].add(frame)
                return attention(*args,**kwargs)
            model.memory_attention.forward=MethodType(attend,model.memory_attention)
            try:
                yield
                if state['captured']!=set(target_frames):raise ValueError('Source capture coverage incomplete')
            finally:model.memory_attention.forward=attention
    finally:
        model._materialize_memory_entry=materialize;model._get_image_feature=get_feature
