"""Read-only MPS early-feature and decoder-head observer for development data.

This captures only the feature/head tensors needed by the refiner; it does not
claim to capture complete memory-attention inputs. The decoder is called once
as in the original propagation and its returned tensors are not changed.
"""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch

from modeling.lean_memory_capture import clone_data


@contextmanager
def observe_decoder_reads(model, targets, output, prototypes, observations):
    if model.device.type != 'mps' or model.fill_hole_area != 0:
        raise ValueError('Observer requires MPS and disabled hole filling')
    if any(name in vars(model) for name in ('_get_image_feature', '_forward_sam_heads')):
        raise RuntimeError('Cannot combine with existing feature/head hooks')
    fg,bg=prototypes
    if fg.shape != (32,) or bg.shape != (32,) or not torch.isfinite(fg).all() or not torch.isfinite(bg).all():
        raise ValueError('Prompt prototype geometry differs')
    old_feature=model._get_image_feature;old_head=model._forward_sam_heads
    state={'frame':None,'video':None,'early':None,'captured':set()}

    def feature(self,inference_state,frame_idx,batch_size):
        result=old_feature(inference_state,frame_idx,batch_size)
        state['frame']=int(frame_idx);state['video']=inference_state
        if state['frame'] in targets and state['frame'] not in state['captured']:
            early=result[1]['backbone_fpn'][0]
            if batch_size!=1 or early.shape!=(1,32,256,256) or not torch.isfinite(early).all():
                raise ValueError('Early feature geometry changed')
            state['early']=early.detach().float().cpu().clone()
        return result

    def head(self,*args,**kwargs):
        if args:raise ValueError('Unexpected positional decoder inputs')
        result=old_head(*args,**kwargs)
        frame=state['frame']
        if frame in targets and frame not in state['captured']:
            early=state['early'];high=kwargs.get('high_res_features')
            if early is None or not high or not torch.equal(early,high[0].detach().float().cpu()):
                raise ValueError('Early feature and decoder input differ')
            if len(result)!=7 or result[3].shape!=(1,1,256,256) or result[4].shape!=(1,1,1024,1024):
                raise ValueError('Decoder output geometry changed')
            root=Path(output);(root/'features').mkdir(exist_ok=True);(root/'snapshots').mkdir(exist_ok=True)
            feature_file=root/'features'/f'{frame:06d}.pt'
            snapshot_file=root/'snapshots'/f'{frame:06d}.pt'
            torch.save({'frame':frame,'feature':early[0],'foreground':fg.clone(),'background':bg.clone()},feature_file)
            torch.save({'frame':frame,'video_height':int(state['video']['video_height']),
                        'video_width':int(state['video']['video_width']),
                        'decoder_inputs':clone_data(kwargs),'decoder_outputs':clone_data(result)},snapshot_file)
            observations.append({'frame':frame,'feature_file':str(feature_file),'snapshot_file':str(snapshot_file)})
            state['captured'].add(frame);state['early']=None
        return result

    model._get_image_feature=MethodType(feature,model)
    model._forward_sam_heads=MethodType(head,model)
    try:
        yield
        if state['captured']!=set(targets):raise ValueError('Observer target coverage differs')
    finally:
        for name in ('_get_image_feature','_forward_sam_heads'):
            if name in vars(model):delattr(model,name)
