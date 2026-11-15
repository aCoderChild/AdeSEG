"""Observe and replace Lean's own fused history segment without feedback.

The author computes the actual reliability and fusion in the normal video
forward. The observer only verifies and saves its existing inputs. Replays can
substitute the corresponding raw skeleton while keeping every other tensor fixed.
"""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import torch
from modeling.lean_memory_capture import capture_memory_read,clone_data
from modeling.lean_region_read_intervention import locate_verified_span


def fused_skeleton(raw,history,alpha):
    if raw.ndim!=3 or history.ndim!=3 or history.shape!=(1,raw.shape[1],raw.shape[2]) or not 0<=alpha<=1:
        raise ValueError('Unexpected fusion geometry')
    return alpha*raw+(1.0-alpha)*history.expand(raw.shape[0],-1,-1)


def replace_fused_skeleton(memory,positions,pointer_count,source,replacement):
    """Replace only feature values in the uniquely verified recorded segment."""
    start=source['start'];raw=source['raw'];fused=source['fused'];n=len(raw);end=start+n
    if memory.ndim!=3 or positions.ndim!=3 or memory.shape[:2]!=positions.shape[:2] or type(pointer_count) is not int or not 0<=pointer_count<=len(memory):raise ValueError('Invalid attention read')
    if type(start) is not int or start<4096 or end>len(memory)-pointer_count or not torch.equal(memory[start:end],fused):raise ValueError('Fused segment no longer matches')
    if replacement not in ('raw','fused'):raise ValueError('Invalid replacement')
    result=memory.clone();result[start:end]=(raw if replacement=='raw' else fused).to(memory.device)
    if pointer_count and not torch.equal(result[-pointer_count:],memory[-pointer_count:]):raise ValueError('Pointer tail changed')
    return result


@contextmanager
def capture_fused_read(model,target_frames,output,observations):
    """Wrap existing methods, without invoking a second model forward."""
    get_feature=model._get_image_feature;materialize=model._materialize_memory_entry;reliability=model._ema_history_reliability
    if model._ema_history_mode()!='fuse_into_latest_old_tail':
        raise ValueError('Author fused history mode differs')
    offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).cpu().clone()
    state={'frame':None,'entries':{},'source':None,'captured':set()}
    def feature(self,inference_state,frame_idx,batch_size):
        if state['frame']!=int(frame_idx):state['entries']={};state['source']=None
        state['frame']=int(frame_idx);return get_feature(inference_state,frame_idx,batch_size)
    def fetch(self,prev,device,*,is_cond=False):
        out=materialize(prev,device,is_cond=is_cond)
        if state['frame'] in target_frames and not is_cond and prev is not None and all(torch.is_tensor(x) for x in out):
            state['entries'][id(prev)]=clone_data(out)
        return out
    def get_reliability(self,prev,ema_state):
        q,info=reliability(prev,ema_state)
        if state['frame'] in target_frames:
            if not self.ema_history_replace_old_tail_in_attn:raise ValueError('Author fused history is not active at read')
            pair=state['entries'].get(id(prev));history=ema_state.get('token')
            if pair is None or not torch.is_tensor(history):raise ValueError('Missing author fusion source or history')
            raw,pos=pair;history=history.detach().cpu().to(dtype=raw.dtype)
            if history.ndim==2:history=history.unsqueeze(0)
            alpha=min(max(float(self.ema_history_fuse_rho_latest)*float(q),0.),1.)
            data={'raw':raw,'spatial_pos':pos,'history':history,'alpha':alpha,'q':float(q),
                  'skeleton_frame':int(prev['frame_idx']),'history_last_frame':ema_state.get('last_absorbed_frame_idx')}
            data['fused']=fused_skeleton(raw,history,alpha)
            if state['source'] is not None:raise ValueError('Multiple author fusion sources in one read')
            state['source']=data
        return q,info
    model._get_image_feature=MethodType(feature,model);model._materialize_memory_entry=MethodType(fetch,model);model._ema_history_reliability=MethodType(get_reliability,model)
    try:
        with capture_memory_read(model,target_frames,output,observations):
            attention=model.memory_attention.forward
            def attend(self,*args,**kwargs):
                frame=state['frame']
                if frame in target_frames and frame not in state['captured']:
                    source=state['source'];memory=kwargs['memory'];pos=kwargs['memory_pos'];ptr=int(kwargs['num_obj_ptr_tokens'])
                    if source is None:raise ValueError('No author fusion source for selected read')
                    start=locate_verified_span(memory,pos,ptr,source['fused'],source['spatial_pos'],offsets,[(0,4096)])
                    source['start']=start;source['frame']=frame;source['pointer_count']=ptr
                    folder=Path(output)/'fusion_sources';folder.mkdir(exist_ok=True);torch.save(source,folder/f'{frame:06d}.pt')
                    state['captured'].add(frame)
                return attention(*args,**kwargs)
            model.memory_attention.forward=MethodType(attend,model.memory_attention)
            try:
                yield
                if state['captured']!=set(target_frames):raise ValueError('Fusion capture coverage incomplete')
            finally:model.memory_attention.forward=attention
    finally:
        model._get_image_feature=get_feature;model._materialize_memory_entry=materialize;model._ema_history_reliability=reliability
