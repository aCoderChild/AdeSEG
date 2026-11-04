import pytest
import torch
from torch import nn

import modeling.decoder_core_training as training
from modeling.decoder_adaptation import train_decoder_core


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(()))
        self.conv_s0=nn.Identity()
        self.conv_s1=nn.Identity()

    def forward(self, value):
        mask=value.reshape(1,1,1,1)*self.weight
        return mask, self.weight.reshape(1,1), None, self.weight.reshape(1,1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder=nn.Identity()
        self.sam_mask_decoder=Decoder()
        self.reads=[]

    def track_step(self, **kwargs):
        bank=kwargs['output_dict']
        entries={**bank['cond_frame_outputs'],**bank['non_cond_frame_outputs']}
        memory=entries[max(entries)]['maskmem_features']
        self.reads.append(float(memory.detach().mean()))
        mask,iou,_,score=self.sam_mask_decoder(kwargs['current_vision_feats'][0]+memory.mean())
        return {'pred_masks':mask,'object_score_logits':score,'obj_ptr':score,
                'maskmem_features':mask,'maskmem_pos_enc':[torch.zeros(1)]}


def samples():
    bank={'cond_frame_outputs':{0:{'maskmem_features':torch.full((1,1,1,1),.25)}},'non_cond_frame_outputs':{}}
    return [{'frame':i,'num_frames':3,'vision':[torch.ones(1)],'positions':[],
             'sizes':[],'output_dict':bank,'target':torch.ones(1,1,1,1)} for i in [1,2]]


def test_clip_uses_predicted_writes_and_preserves_native_prefix(monkeypatch):
    def loss(captured,target):
        value=(captured[0][0]-target).square().mean()
        return value,{'core_loss':value}
    monkeypatch.setattr(training,'decoder_loss',loss)
    model=Model()
    train_decoder_core(model)
    batch=samples()
    value,info=training.decoder_clip_loss(model,batch,retain_first_write=True)
    value.backward()
    assert model.reads==[.25,1.25]
    assert info['first_write'].grad.abs().sum()>0
    assert torch.isfinite(model.sam_mask_decoder.weight.grad)
    assert batch[0]['output_dict']['non_cond_frame_outputs']=={}
    assert batch[0]['output_dict']['cond_frame_outputs'][0]['maskmem_features'].item()==.25


def test_clip_rejects_nonconsecutive_targets_before_model_read():
    model=Model()
    train_decoder_core(model)
    batch=samples()
    batch[1]['frame']=3
    with pytest.raises(ValueError,match='consecutive'):
        training.decoder_clip_loss(model,batch)
    assert model.reads==[]
