from pathlib import Path
import torch
from modeling.lean_memory_encoder_capture import capture_memory_encoder

class Dummy:
    def _get_image_feature(self,inference_state,frame_idx,batch_size):return frame_idx
    def _encode_new_memory(self,**kw):return (kw['x']*2,[kw['x']+1])

def test_capture_is_observational_and_restores_methods(tmp_path: Path):
    model=Dummy();events=[];x=torch.tensor([3.])
    with capture_memory_encoder(model,{7},tmp_path,events):
        assert model._get_image_feature({},7,1)==7
        result=model._encode_new_memory(x=x)
        assert torch.equal(result[0],torch.tensor([6.]))
    snap=torch.load(tmp_path/'encoder_snapshots/000007.pt',weights_only=True)
    assert snap['frame']==7 and torch.equal(snap['inputs']['x'],x) and torch.equal(snap['outputs'][0],result[0])
    assert model._get_image_feature({},8,1)==8
    assert torch.equal(model._encode_new_memory(x=x)[0],result[0])
    assert len(events)==1
