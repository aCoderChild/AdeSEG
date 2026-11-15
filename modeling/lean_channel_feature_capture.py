"""Observe early SAM2 features and automatic prompt prototypes without inference changes."""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import hashlib
import torch
from modeling.prompt_feature_refinement import reference_prototypes


def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@contextmanager
def capture_channel_features(model,target_frames,output,observations):
    get_feature=model._get_image_feature;add_mask=model.add_new_mask
    state={'frame':None,'feature':None,'prototypes':None,'captured':set()}
    def feature(self,inference_state,frame_idx,batch_size):
        out=get_feature(inference_state,frame_idx,batch_size);f=out[1]['backbone_fpn'][0]
        if batch_size!=1 or f.ndim!=4 or f.shape!=(1,32,256,256) or f.device.type!='cpu' or not torch.isfinite(f).all():
            raise ValueError('Early FPN interface changed')
        frame=int(frame_idx);state['frame']=frame;state['feature']=f[0].detach().float()
        if frame in target_frames and frame not in state['captured']:
            if state['prototypes'] is None:raise ValueError('Feature captured before automatic initial prompt')
            fg,bg=state['prototypes'];sample={'frame':frame,'feature':state['feature'].cpu().clone(),'foreground':fg.cpu().clone(),'background':bg.cpu().clone()}
            folder=Path(output)/'features';folder.mkdir(exist_ok=True);torch.save(sample,folder/f'{frame:06d}.pt')
            observations.append({'frame':frame,'feature_sha256':sha(sample['feature']),'foreground_sha256':sha(sample['foreground']),'background_sha256':sha(sample['background'])})
            state['captured'].add(frame)
        return out
    def mask(self,inference_state,frame_idx,obj_id,mask):
        if state['frame']!=int(frame_idx) or state['prototypes'] is not None:raise ValueError('Automatic prompt source differs')
        state['prototypes']=reference_prototypes(state['feature'],mask)
        return add_mask(inference_state,frame_idx=frame_idx,obj_id=obj_id,mask=mask)
    model._get_image_feature=MethodType(feature,model);model.add_new_mask=MethodType(mask,model)
    try:
        yield
        if state['captured']!=set(target_frames):raise ValueError('Feature capture coverage incomplete')
    finally:model._get_image_feature=get_feature;model.add_new_mask=add_mask
