"""Read-only capture of the active Lean memory encoder's exact inputs/outputs."""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from modeling.lean_memory_capture import clone_data
import torch

@contextmanager
def capture_memory_encoder(model,target_frames,output,observations):
    get_feature=model._get_image_feature;encode=model._encode_new_memory
    state={'frame':None,'captured':set()}
    def feature(self,inference_state,frame_idx,batch_size):
        result=get_feature(inference_state,frame_idx,batch_size)
        state['frame']=int(frame_idx)
        return result
    def encoder(self,*args,**kwargs):
        if args:raise ValueError('Unexpected positional memory-encoder arguments')
        frame=state['frame'];capture=frame in target_frames
        if capture and frame in state['captured']:raise ValueError('Repeated target-frame memory encoding')
        inputs=clone_data(kwargs) if capture else None
        result=encode(**kwargs)
        if capture:
            snapshot={'frame':frame,'inputs':inputs,'outputs':clone_data(result)}
            folder=Path(output)/'encoder_snapshots';folder.mkdir(exist_ok=True)
            path=folder/f'{frame:06d}.pt';torch.save(snapshot,path)
            observations.append({'frame':frame,'snapshot_file':path.name})
            state['captured'].add(frame)
        return result
    model._get_image_feature=MethodType(feature,model);model._encode_new_memory=MethodType(encoder,model)
    try:
        yield
        if state['captured']!=set(target_frames):raise ValueError('Encoder capture scope incomplete')
    finally:
        model._get_image_feature=get_feature;model._encode_new_memory=encode
