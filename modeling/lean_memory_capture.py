"""Read-only snapshots of author attention and decoder inputs and outputs."""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import torch


def clone_data(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:clone_data(v) for k,v in value.items()}
    if isinstance(value,list):
        return [clone_data(v) for v in value]
    if isinstance(value,tuple):
        return tuple(clone_data(v) for v in value)
    if value is None or isinstance(value,(bool,int,float,str)):
        return value
    raise ValueError(f'Unsupported snapshot value: {type(value)}')


def zero_memory_segment(memory,pointer_count,segment):
    """Clone the spatial values to ablate; preserve pointer-tail values exactly."""
    if memory.ndim!=3 or memory.shape[1]!=1 or not 0<=pointer_count<=memory.shape[0]-4096:
        raise ValueError('Unexpected memory geometry')
    end=memory.shape[0]-pointer_count
    changed=memory.clone()
    if segment=='conditioning':changed[:4096]=0
    elif segment=='other_spatial':changed[4096:end]=0
    elif segment!='noop':raise ValueError('Unknown shadow segment')
    if not torch.equal(changed[end:],memory[end:]):raise ValueError('Pointer tail changed')
    return changed


@contextmanager
def capture_memory_read(model,target_frames,output,observations):
    """Capture author tensors; no extra model call is executed in propagation."""
    if model.fill_hole_area!=0:raise ValueError('Snapshot requires disabled hole filling')
    get_feature=model._get_image_feature;attention=model.memory_attention.forward;head=model._forward_sam_heads
    with torch.no_grad():
        grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1).contiguous()
        expected=grid+model.maskmem_tpos_enc[model.num_maskmem-1].detach()
    state={'frame':None,'video':None,'pending':None,'captured':set()}
    def feature(self,inference_state,frame_idx,batch_size):
        result=get_feature(inference_state,frame_idx,batch_size);state['frame']=int(frame_idx);state['video']=inference_state;return result
    def attend(self,*args,**kwargs):
        if args:raise ValueError('Unexpected attention arguments')
        frame=state['frame'];capture=frame in target_frames and frame not in state['captured']
        if capture:
            if state['pending'] is not None or not kwargs.get('return_attn_stat'):raise ValueError('Unexpected pending snapshot')
            memory=kwargs['memory'];pos=kwargs['memory_pos'];ptr=int(kwargs['num_obj_ptr_tokens'])
            if memory.ndim!=3 or memory.shape[1]!=1 or memory.shape[0]!=pos.shape[0] or not torch.equal(pos[:4096],expected):raise ValueError('Conditioning segment differs')
            captured_inputs=clone_data(kwargs)
        result=attention(*args,**kwargs)
        if capture:
            state['pending']={'frame':frame,'attention_inputs':captured_inputs,'attention_outputs':clone_data(result),
                              'video_height':int(state['video']['video_height']),'video_width':int(state['video']['video_width'])}
        return result
    def decode(self,*args,**kwargs):
        pending=state['pending'];captured_inputs=clone_data(kwargs) if pending is not None else None
        if pending is not None and args:raise ValueError('Unexpected decoder positional arguments')
        result=head(*args,**kwargs)
        if pending is not None:
            x=pending['attention_outputs'][0]
            if x.shape[0]!=4096 or x.shape[1]!=1 or not torch.equal(captured_inputs['backbone_features'],x.permute(1,2,0).view(1,x.shape[2],64,64)):
                raise ValueError('Captured decoder input does not match attention output')
            pending['decoder_inputs']=captured_inputs;pending['decoder_outputs']=clone_data(result)
            folder=Path(output)/'snapshots';folder.mkdir(exist_ok=True);path=folder/f"{pending['frame']:06d}.pt"
            torch.save(pending,path)
            observations.append({'frame':pending['frame'],'snapshot_file':path.name})
            state['captured'].add(pending['frame']);state['pending']=None
        return result
    model._get_image_feature=MethodType(feature,model);model.memory_attention.forward=MethodType(attend,model.memory_attention);model._forward_sam_heads=MethodType(decode,model)
    try:
        yield
        if state['pending'] is not None or state['captured']!=set(target_frames):raise ValueError('Snapshot coverage incomplete')
    finally:
        model._get_image_feature=get_feature;model.memory_attention.forward=attention;model._forward_sam_heads=head
