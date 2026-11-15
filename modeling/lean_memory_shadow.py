"""Quarantined in-process shadow experiment; real-video parity failed.

Retained for investigation only. Do not use its masks or run it in a video.
"""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import hashlib
import numpy as np
from PIL import Image
import torch


@contextmanager
def memory_shadow_observer(model, positions, target_frames, output, observations):
    """Reject an observer known to change later real-video predictions."""
    raise RuntimeError('In-process shadow observer is disabled: real-video replay parity failed')
    if model.fill_hole_area != 0:
        raise ValueError('Shadow mask comparison requires disabled hole filling')
    get_feature=model._get_image_feature
    attention=model.memory_attention.forward
    head=model._forward_sam_heads
    with torch.no_grad():
        grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1).contiguous()
        expected=grid+model.maskmem_tpos_enc[model.num_maskmem-1].detach()
    state={'frame':None,'video':None,'pending':None,'captured':set()}

    def feature(self,inference_state,frame_idx,batch_size):
        result=get_feature(inference_state,frame_idx,batch_size)
        state['frame']=int(frame_idx);state['video']=inference_state
        return result

    def attend(self,*args,**kwargs):
        baseline=attention(*args,**kwargs)
        frame=state['frame']
        if frame not in target_frames or frame in state['captured']:
            return baseline
        if args or not kwargs.get('return_attn_stat') or state['pending'] is not None:
            raise ValueError('Unexpected attention invocation or pending shadow')
        memory=kwargs['memory'];pos=kwargs['memory_pos'];ptr=int(kwargs['num_obj_ptr_tokens'])
        if memory.ndim!=3 or memory.shape[1]!=1 or memory.shape[0]!=pos.shape[0] or not torch.equal(pos[:4096],expected):
            raise ValueError('Conditioning segment mismatch')
        end=memory.shape[0]-ptr
        if end<4096:
            raise ValueError('Invalid pointer tail')
        retained_events=len(positions)
        original_inputs={k:v.detach().clone() for k,v in kwargs.items() if torch.is_tensor(v)}
        original_result=[x.detach().clone() for x in baseline]
        original_rng=torch.get_rng_state().clone()
        shadows={}
        for name in ('noop','conditioning_zero','other_spatial_zero'):
            altered=memory.clone()
            if name=='conditioning_zero':altered[:4096]=0
            if name=='other_spatial_zero':altered[4096:end]=0
            if not torch.equal(altered[end:],memory[end:]) or (name=='conditioning_zero' and torch.count_nonzero(altered[:4096])) or (name=='other_spatial_zero' and torch.count_nonzero(altered[4096:end])):
                raise ValueError('Shadow altered the wrong memory segment')
            shadow_kwargs=dict(kwargs);shadow_kwargs['memory']=altered
            shadow=attention(**shadow_kwargs)
            if not torch.equal(original_rng,torch.get_rng_state()):
                raise ValueError(f'{name}: shadow changed RNG state')
            if any(not torch.equal(original_inputs[k],kwargs[k]) for k in original_inputs):
                raise ValueError(f'{name}: shadow mutated live attention inputs')
            if any(not torch.equal(x,y) for x,y in zip(original_result,baseline)):
                raise ValueError(f'{name}: shadow mutated live attention outputs')
            # The indexed adapter records every call; shadow-only entries are not
            # part of the live trajectory and are removed from its receipt.
            if len(positions)!=retained_events+1:
                raise ValueError('Unexpected indexed shadow trace')
            del positions[retained_events:]
            if name=='noop' and (not torch.equal(shadow[0],baseline[0]) or not torch.equal(shadow[1],baseline[1])):
                raise ValueError('Unchanged-memory shadow differs')
            shadows[name]=shadow[0]
        state['pending']={'frame':frame,'baseline':baseline[0],'shadows':shadows,'other_spatial_tokens':int(end-4096),'pointer_tokens':ptr}
        return baseline

    def decode(self,*args,**kwargs):
        original=head(*args,**kwargs)
        pending=state['pending']
        if pending is None:
            return original
        if args or state['video'] is None or 'backbone_features' not in kwargs:
            raise ValueError('Unexpected decoder invocation')
        x=pending['baseline']
        if x.ndim!=3 or x.shape[0]!=4096 or x.shape[1]!=1:
            raise ValueError('Unexpected memory-attention feature shape')
        baseline_feature=x.permute(1,2,0).view(1,x.shape[2],64,64)
        if not torch.equal(kwargs['backbone_features'],baseline_feature):
            raise ValueError('Decoder input does not match observed memory output')
        choices={'baseline':original}
        original_inputs={k:(v.detach().clone() if torch.is_tensor(v) else [x.detach().clone() for x in v] if isinstance(v,list) and all(torch.is_tensor(x) for x in v) else None) for k,v in kwargs.items()}
        original_result=[x.detach().clone() for x in original]
        original_rng=torch.get_rng_state().clone()
        for name,tensor in pending['shadows'].items():
            alternative=dict(kwargs);alternative['backbone_features']=tensor.permute(1,2,0).view(1,tensor.shape[2],64,64)
            choices[name]=head(**alternative)
            if not torch.equal(original_rng,torch.get_rng_state()):
                raise ValueError(f'{name}: shadow decoder changed RNG state')
            for k,v in original_inputs.items():
                if v is None:continue
                if isinstance(v,list):
                    if any(not torch.equal(x,y) for x,y in zip(v,kwargs[k])):raise ValueError(f'{name}: shadow decoder mutated {k}')
                elif not torch.equal(v,kwargs[k]):raise ValueError(f'{name}: shadow decoder mutated {k}')
            if any(not torch.equal(x,y) for x,y in zip(original_result,original)):
                raise ValueError(f'{name}: shadow decoder mutated live output')
        if any(not torch.equal(x,y) for x,y in zip(original,choices['noop'])):
            raise ValueError('Unchanged-memory shadow decoder differs')
        folder=Path(output)/'shadow';folder.mkdir(exist_ok=True)
        files={}
        for name,result in choices.items():
            _,video=self._get_orig_video_res_output(state['video'],result[3])
            binary=(video[0,0].detach().cpu().numpy()>0).astype(np.uint8)*255
            path=folder/f"{pending['frame']:06d}_{name}.png";Image.fromarray(binary).save(path)
            files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        observations.append({'frame':pending['frame'],'other_spatial_tokens':pending['other_spatial_tokens'],
                             'pointer_tokens':pending['pointer_tokens'],'mask_sha256':files})
        state['captured'].add(pending['frame']);state['pending']=None
        return original

    model._get_image_feature=MethodType(feature,model)
    model.memory_attention.forward=MethodType(attend,model.memory_attention)
    model._forward_sam_heads=MethodType(decode,model)
    try:
        yield
        if state['pending'] is not None or state['captured']!=set(target_frames):
            raise ValueError('Shadow capture incomplete')
    finally:
        model._get_image_feature=get_feature
        model.memory_attention.forward=attention
        model._forward_sam_heads=head
