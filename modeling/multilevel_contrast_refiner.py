"""Trainable multi-level mask-logit refiner using the pinned TCMN channel gate.

Research prototype only: a scoped Lean predictor adapter exists, but the
refiner has not been trained or validated for video experiments. The author gate
is reused; alignment, soft contrast and residual head are project-specific.
An untrained module cannot be evaluated in inference mode.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.lean_channel_attention import author_sha256, load_author_channel_gate


class MultilevelContrastRefiner(nn.Module):
    def __init__(self, early_channels=32, deep_channels=256, hidden=64):
        super().__init__()
        if hidden < 16 or hidden % 16:
            raise ValueError('Hidden channels must be a positive multiple of 16')
        self.early_channels=early_channels;self.deep_channels=deep_channels
        self.early=nn.Conv2d(early_channels,hidden//2,1)
        self.deep=nn.Conv2d(deep_channels,hidden//2,1)
        self.gate=load_author_channel_gate()(hidden,reduction_ratio=16,pool_types=['avg','max'])
        self.head=nn.Sequential(nn.Conv2d(hidden+2,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,1,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self._trained_checkpoint_loaded=False
        self._mask_only=False

    def load_trained_state(self, payload, *, backbone_sha256, feature_pipeline_sha256, train_manifest_sha256):
        # Any failed reload must revoke prior inference authorization: strict
        # loading can mutate parameters before reporting an error.
        self._trained_checkpoint_loaded=False
        self._mask_only=False
        if not isinstance(payload,dict) or payload.get('format')!='multilevel_contrast_refiner_v1' or type(payload.get('training_steps')) is not int or payload['training_steps']<=0 or payload.get('research_only') is not True:
            raise ValueError('Trained, research-only checkpoint metadata required')
        if payload.get('author_sha256')!=author_sha256():
            raise ValueError('Checkpoint author-gate implementation differs')
        for key, expected in (('backbone_sha256',backbone_sha256),('feature_pipeline_sha256',feature_pipeline_sha256),('train_manifest_sha256',train_manifest_sha256)):
            if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected) or payload.get(key)!=expected:
                raise ValueError(f'Checkpoint provenance mismatch: {key}')
        arm=payload.get('arm','multilevel')
        if arm not in ('multilevel','mask_only'):
            raise ValueError('Unknown checkpoint arm')
        self.load_state_dict(payload['state_dict'],strict=True)
        if any(not torch.isfinite(p).all() for p in self.parameters()):raise ValueError('Nonfinite checkpoint')
        self._mask_only=(arm=='mask_only')
        self._trained_checkpoint_loaded=True

    def forward(self, early, deep, logits):
        if (not self.training or not torch.is_grad_enabled()) and not self._trained_checkpoint_loaded:
            raise RuntimeError('Untrained refinement cannot be used in evaluation')
        if early.ndim!=4 or deep.ndim!=4 or logits.ndim!=4 or logits.shape[1]!=1 or early.shape[1]!=self.early_channels or deep.shape[1]!=self.deep_channels or early.shape[0]!=deep.shape[0] or early.shape[0]!=logits.shape[0]:
            raise ValueError('Feature or mask geometry differs')
        if early.device!=deep.device or early.device!=logits.device or not all(torch.isfinite(t).all() for t in (early,deep,logits)):
            raise ValueError('Inputs must be finite and on one device')
        if self._mask_only:
            early=torch.zeros_like(early);deep=torch.zeros_like(deep)
        size=logits.shape[-2:]
        e=F.interpolate(self.early(early.float()),size=size,mode='bilinear',align_corners=False)
        d=F.interpolate(self.deep(deep.float()),size=size,mode='bilinear',align_corners=False)
        features=torch.cat((e,d),dim=1);prob=torch.sigmoid(logits.float())
        # Soft averages avoid hard region thresholds and remain finite for empty masks.
        fg=(features*prob).sum((-2,-1),keepdim=True)/prob.sum((-2,-1),keepdim=True).clamp_min(1e-6)
        bg=(features*(1-prob)).sum((-2,-1),keepdim=True)/(1-prob).sum((-2,-1),keepdim=True).clamp_min(1e-6)
        contrast=F.cosine_similarity(features,fg,dim=1,eps=1e-6)-F.cosine_similarity(features,bg,dim=1,eps=1e-6)
        weighted=self.gate(features)
        residual=self.head(torch.cat((weighted,contrast[:,None],logits.float()),dim=1))
        return logits+residual.to(dtype=logits.dtype)
