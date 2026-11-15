"""Experimental continuous use of native raw decoder candidates as features.

Raw candidate masks and predicted IoUs come from the existing author decoder
call. The refiner neither changes native object scores/pointers nor adds a
hard gate, detector, threshold or pruning rule. Its continuous final residual
could nevertheless overcome a native -1024 mask: this would relax output mask
suppression and must not be presented as preserving the original mask decision.
This prototype is not authorized for live video use.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.foreground_conditioned_refiner import ForegroundConditionedRefiner,_source_sha256
from modeling.lean_channel_attention import author_sha256


class RawMultimaskRefiner(ForegroundConditionedRefiner):
    """Add raw-candidate features without changing the native output base.

    All controls have identical learnable parameters. `zero` removes raw input;
    `spatial_shift` rolls candidate maps but keeps their global IoU values.
    The shift tests spatial alignment, not all candidate information.
    """

    def __init__(self, *, raw_mode='raw', gate_mode='region', contrast_head=True, **kwargs):
        super().__init__(gate_mode=gate_mode,contrast_head=contrast_head,**kwargs)
        if raw_mode not in ('raw','zero','spatial_shift'):
            raise ValueError('Unsupported raw-candidate ablation')
        self.raw_mode=raw_mode
        hidden=self.early.out_channels+self.deep.out_channels
        self.raw_adapter=nn.Conv2d(6,hidden,1)
        nn.init.zeros_(self.raw_adapter.weight);nn.init.zeros_(self.raw_adapter.bias)

    def load_trained_state(self,payload,*,backbone_sha256,feature_pipeline_sha256,train_manifest_sha256,raw_manifest_sha256):
        self._trained_checkpoint_loaded=False;self._mask_only=False
        if not isinstance(payload,dict) or payload.get('format')!='raw_multimask_refiner_v1' or payload.get('research_only') is not True or type(payload.get('training_steps')) is not int or payload['training_steps']<=0:
            raise ValueError('Matching trained raw-multimask checkpoint required')
        files=(('raw_model_sha256',__file__),('conditioned_source_sha256',Path(__file__).with_name('foreground_conditioned_refiner.py')),
               ('base_source_sha256',Path(__file__).with_name('multilevel_contrast_refiner.py')))
        if payload.get('author_sha256')!=author_sha256() or any(payload.get(k)!=_source_sha256(path) for k,path in files):
            raise ValueError('Raw refiner, parent refiner or author source changed')
        if payload.get('raw_mode')!=self.raw_mode or payload.get('gate_mode')!=self.gate_mode or payload.get('contrast_head') is not self.contrast_head:
            raise ValueError('Checkpoint ablation differs')
        for key,expected in (('backbone_sha256',backbone_sha256),('feature_pipeline_sha256',feature_pipeline_sha256),
                             ('train_manifest_sha256',train_manifest_sha256),('raw_manifest_sha256',raw_manifest_sha256)):
            if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected) or payload.get(key)!=expected:
                raise ValueError(f'Checkpoint provenance mismatch: {key}')
        self.load_state_dict(payload['state_dict'],strict=True)
        if any(not torch.isfinite(p).all() for p in self.parameters()):raise ValueError('Nonfinite checkpoint')
        self._trained_checkpoint_loaded=True

    def raw_input(self,raw_candidates,predicted_ious):
        maps=torch.tanh(raw_candidates.float()/16.0)
        if self.raw_mode=='spatial_shift':maps=maps.roll(shifts=97,dims=-1)
        ious=predicted_ious.float()[:,:,None,None].expand_as(maps)
        context=torch.cat((maps,ious),dim=1)
        return torch.zeros_like(context) if self.raw_mode=='zero' else context

    def forward(self,early,deep,logits,raw_candidates=None,predicted_ious=None):
        if raw_candidates is None or predicted_ious is None:
            raise RuntimeError('Raw multimask inputs are required; the existing live router does not support this model')
        if (not self.training or not torch.is_grad_enabled()) and not self._trained_checkpoint_loaded:
            raise RuntimeError('Untrained refinement cannot be used in evaluation')
        if (early.ndim!=4 or deep.ndim!=4 or logits.ndim!=4 or logits.shape[1]!=1
                or early.shape[1]!=self.early_channels or deep.shape[1]!=self.deep_channels
                or early.shape[0]!=deep.shape[0] or early.shape[0]!=logits.shape[0]
                or raw_candidates.shape!=(logits.shape[0],3,*logits.shape[-2:])
                or predicted_ious.shape!=(logits.shape[0],3)):
            raise ValueError('Feature, mask or raw-candidate geometry differs')
        if any(x.device!=logits.device for x in (early,deep,raw_candidates,predicted_ious)) or not all(torch.isfinite(x).all() for x in (early,deep,logits,raw_candidates,predicted_ious)):
            raise ValueError('Inputs must be finite and on one device')
        size=logits.shape[-2:]
        e=F.interpolate(self.early(early.float()),size=size,mode='bilinear',align_corners=False)
        d=F.interpolate(self.deep(deep.float()),size=size,mode='bilinear',align_corners=False)
        features=torch.cat((e,d),dim=1)+self.raw_adapter(self.raw_input(raw_candidates,predicted_ious))
        prob=logits.float().sigmoid()
        fg=(features*prob).sum((-2,-1),keepdim=True)/prob.sum((-2,-1),keepdim=True).clamp_min(1e-6)
        bg=(features*(1-prob)).sum((-2,-1),keepdim=True)/(1-prob).sum((-2,-1),keepdim=True).clamp_min(1e-6)
        contrast=F.cosine_similarity(features,fg,dim=1,eps=1e-6)-F.cosine_similarity(features,bg,dim=1,eps=1e-6)
        if not self.contrast_head:contrast=torch.zeros_like(contrast)
        weighted=features*self.channel_weights(features,fg,bg)
        residual=self.head(torch.cat((weighted,contrast[:,None],logits.float()),dim=1))
        return logits+residual.to(dtype=logits.dtype)
