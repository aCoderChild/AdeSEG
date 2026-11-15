"""Guarded experimental adapter around the TCMN author's CBAM channel gate.

The gate is trained on two region descriptors [foreground, -background] in the
SAM2 early-FPN space. Its 32-channel weights must be trained for that space;
TCMN's 512/1024-channel checkpoints are not interchangeable. This adapter does
not implement the full TCMN model or claim zero-shot pretrained performance.
"""
from pathlib import Path
import hashlib
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
AUTHOR=ROOT/'external/TCMN/model/cbam.py'


def author_sha256():
    return hashlib.sha256(AUTHOR.read_bytes()).hexdigest()


def load_author_channel_gate():
    spec=importlib.util.spec_from_file_location('_tcmn_author_cbam',AUTHOR)
    if spec is None or spec.loader is None:raise RuntimeError('TCMN author CBAM unavailable')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.ChannelGate


class PromptContrastChannelAttention(nn.Module):
    """Region-conditioned channel weights for aligned feature contrast.

    A checkpoint is mandatory for evaluation; training must explicitly opt in
    to initialization. No confidence gate, re-detection or GT-derived inference
    input is added. Inputs are foreground and background prototype vectors from
    the initial automatic mask and a matching source-frame early feature grid.
    """
    def __init__(self,channels=32,checkpoint=None,*,initialize_for_training=False,expected_backbone_sha256=None,expected_feature_pipeline_sha256=None):
        super().__init__()
        if type(channels) is not int or channels!=32:raise ValueError('The verified early-FPN interface requires 32 channels')
        if checkpoint is None and not initialize_for_training:raise ValueError('Trained checkpoint required for inference')
        if checkpoint is not None and initialize_for_training:raise ValueError('Choose checkpoint or training initialization')
        self.channels=channels
        self.gate=load_author_channel_gate()(channels,reduction_ratio=16,pool_types=['avg','max'])
        self.checkpoint_metadata=None
        if checkpoint is not None:
            data=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
            if not isinstance(data,dict) or data.get('format')!='sam2_early_prompt_cbam_v1' or data.get('channels')!=channels or data.get('author_sha256')!=author_sha256() or type(data.get('training_steps')) is not int or data['training_steps']<=0 or data.get('feature_interface')!='sam2.1_hiera_base_plus.conv_s0.fpn0':
                raise ValueError('Checkpoint lacks matching author/interface/training metadata')
            if data.get('engineering_only',False) is not False:
                raise ValueError('Engineering fixture cannot be used for inference')
            if not isinstance(expected_backbone_sha256,str) or len(expected_backbone_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_backbone_sha256) or data.get('backbone_sha256')!=expected_backbone_sha256:
                raise ValueError('Checkpoint does not match the pinned SAM2 backbone')
            if not isinstance(expected_feature_pipeline_sha256,str) or len(expected_feature_pipeline_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_feature_pipeline_sha256) or data.get('feature_pipeline_sha256')!=expected_feature_pipeline_sha256:
                raise ValueError('Checkpoint does not match the pinned feature pipeline')
            train_sha=data.get('train_manifest_sha256')
            if not isinstance(train_sha,str) or len(train_sha)!=64 or any(c not in '0123456789abcdef' for c in train_sha):
                raise ValueError('Checkpoint lacks a pinned training manifest')
            self.gate.load_state_dict(data['gate_state_dict'],strict=True)
            if any(not torch.isfinite(p).all() for p in self.gate.parameters()):
                raise ValueError('Checkpoint contains nonfinite gate weights')
            self.checkpoint_metadata={k:v for k,v in data.items() if k!='gate_state_dict'}

    def channel_weights(self,foreground,background):
        if foreground.shape!=(self.channels,) or background.shape!=(self.channels,) or not torch.isfinite(foreground).all() or not torch.isfinite(background).all():
            raise ValueError('Expected finite, aligned region descriptors')
        if self.checkpoint_metadata is None and (not self.training or not torch.is_grad_enabled()):
            raise RuntimeError('Untrained attention cannot run in inference mode')
        descriptor=torch.stack((foreground,-background),dim=-1)[None,:,None,:]
        outputs=[]
        hook=self.gate.mlp.register_forward_hook(lambda _m,_i,o:outputs.append(o))
        try:gated=self.gate(descriptor)
        finally:hook.remove()
        if len(outputs)!=2:raise RuntimeError('Author gate pooling path changed')
        weights=torch.sigmoid(outputs[0]+outputs[1]).reshape(self.channels)
        # Ensure recovered weights exactly reproduce the author's gate.
        if not torch.equal(gated,descriptor*weights[None,:,None,None]):raise RuntimeError('Author gate response differs')
        return weights

    def forward(self,feature,foreground,background,spatial_size=(64,64),*,gate_foreground=None,gate_background=None):
        if feature.ndim!=3 or feature.shape[0]!=self.channels or not torch.isfinite(feature).all():raise ValueError('Expected matching early-FPN feature grid')
        if foreground.device!=feature.device or background.device!=feature.device:raise ValueError('Feature and prototypes must share a device')
        if (gate_foreground is None)!=(gate_background is None):raise ValueError('Gate descriptors must be provided together')
        if gate_foreground is None:gate_foreground,gate_background=foreground,background
        if gate_foreground.device!=feature.device or gate_background.device!=feature.device:raise ValueError('Gate descriptors must share the feature device')
        weights=self.channel_weights(gate_foreground,gate_background)
        f=F.normalize(feature.float()*weights[:,None,None],dim=0)
        fg=F.normalize(foreground.float()*weights,dim=0);bg=F.normalize(background.float()*weights,dim=0)
        similarity=(torch.einsum('chw,c->hw',f,fg)-torch.einsum('chw,c->hw',f,bg))[None,None]*.5
        score=F.interpolate(similarity,size=spatial_size,mode='area').reshape(-1)
        if not torch.isfinite(score).all():raise ValueError('Nonfinite contrast score')
        return score,weights
