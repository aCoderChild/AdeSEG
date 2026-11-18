"""Research candidate: predicted-region-conditioned CBAM channel reweighting.

This adapts the TCMN author's CBAM gate. Its shared MLP mixes information
between channels when computing weights; the output operation reweights
channels and does not mix their values. The subsequent residual head and
soft foreground/background cosine contrast come from the project refiner.
This prototype is not authorized for live video propagation.
"""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from modeling.lean_channel_attention import author_sha256
from modeling.multilevel_contrast_refiner import MultilevelContrastRefiner


def _source_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ForegroundConditionedRefiner(MultilevelContrastRefiner):
    """Compare channel-weight conditioning without changing model capacity.

    ``region`` computes weights from soft foreground and negated-background
    descriptors; ``global`` uses the author's full-image average/max pooling;
    ``shuffled`` rolls region descriptor channels before the same author MLP.
    Every mode uses the same parameters and head. Foreground/background weights
    are computed from automatic mask logits, not labels or thresholding.
    """

    def __init__(self, *, gate_mode='region', contrast_head=True, **kwargs):
        super().__init__(**kwargs)
        if gate_mode not in ('region', 'global', 'shuffled') or type(contrast_head) is not bool:
            raise ValueError('Unsupported attention ablation')
        self.gate_mode = gate_mode
        self.contrast_head = contrast_head

    def channel_weights(self, features, foreground, background):
        """Use the author's average/max pooled shared MLP on the chosen input."""
        if features.ndim != 4 or foreground.shape != features.shape[:2]+(1,1) or background.shape != foreground.shape:
            raise ValueError('Channel descriptor geometry differs')
        if self.gate_mode == 'global':
            gate_input = features
        else:
            gate_input = torch.cat((foreground, -background), dim=-1)
            if self.gate_mode == 'shuffled':
                gate_input = gate_input.roll(shifts=1, dims=1)
        # These are the exact pool operations and shared author MLP used by
        # ChannelGate.forward; no fresh randomly initialized attention block.
        if self.gate.pool_types != ['avg', 'max']:
            raise ValueError('Author gate pooling differs')
        size = gate_input.shape[-2:]
        avg = F.avg_pool2d(gate_input, size, stride=size)
        maximum = F.max_pool2d(gate_input, size, stride=size)
        return torch.sigmoid(self.gate.mlp(avg) + self.gate.mlp(maximum))[:, :, None, None]

    def load_trained_state(self, payload, *, backbone_sha256, feature_pipeline_sha256, train_manifest_sha256):
        # Do not accept an earlier non-conditioned checkpoint as this method.
        self._trained_checkpoint_loaded = False
        self._mask_only = False
        if not isinstance(payload, dict) or payload.get('format') != 'foreground_conditioned_refiner_v1' or payload.get('research_only') is not True or type(payload.get('training_steps')) is not int or payload['training_steps'] <= 0:
            raise ValueError('Matching trained research checkpoint required')
        if payload.get('author_sha256') != author_sha256() or payload.get('conditioned_source_sha256') != _source_sha256(__file__) or payload.get('base_source_sha256') != _source_sha256(Path(__file__).with_name('multilevel_contrast_refiner.py')):
            raise ValueError('Refiner or author-gate source changed')
        if payload.get('gate_mode') != self.gate_mode or payload.get('contrast_head') is not self.contrast_head:
            raise ValueError('Checkpoint attention ablation differs')
        for key, expected in (('backbone_sha256', backbone_sha256), ('feature_pipeline_sha256', feature_pipeline_sha256), ('train_manifest_sha256', train_manifest_sha256)):
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected) or payload.get(key) != expected:
                raise ValueError(f'Checkpoint provenance mismatch: {key}')
        arm = payload.get('arm')
        if arm not in ('multilevel', 'mask_only'):
            raise ValueError('Unknown checkpoint input arm')
        self.load_state_dict(payload['state_dict'], strict=True)
        if any(not torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError('Nonfinite checkpoint')
        self._mask_only = arm == 'mask_only'
        self._trained_checkpoint_loaded = True

    def forward(self, early, deep, logits):
        if (not self.training or not torch.is_grad_enabled()) and not self._trained_checkpoint_loaded:
            raise RuntimeError('Untrained refinement cannot be used in evaluation')
        if early.ndim != 4 or deep.ndim != 4 or logits.ndim != 4 or logits.shape[1] != 1 or early.shape[1] != self.early_channels or deep.shape[1] != self.deep_channels or early.shape[0] != deep.shape[0] or early.shape[0] != logits.shape[0]:
            raise ValueError('Feature or mask geometry differs')
        if early.device != deep.device or early.device != logits.device or not all(torch.isfinite(t).all() for t in (early, deep, logits)):
            raise ValueError('Inputs must be finite and on one device')
        if self._mask_only:
            early = torch.zeros_like(early)
            deep = torch.zeros_like(deep)
        size = logits.shape[-2:]
        e = F.interpolate(self.early(early.float()), size=size, mode='bilinear', align_corners=False)
        d = F.interpolate(self.deep(deep.float()), size=size, mode='bilinear', align_corners=False)
        features = torch.cat((e, d), dim=1)
        prob = torch.sigmoid(logits.float())
        fg = (features * prob).sum((-2, -1), keepdim=True) / prob.sum((-2, -1), keepdim=True).clamp_min(1e-6)
        bg = (features * (1-prob)).sum((-2, -1), keepdim=True) / (1-prob).sum((-2, -1), keepdim=True).clamp_min(1e-6)
        contrast = F.cosine_similarity(features, fg, dim=1, eps=1e-6) - F.cosine_similarity(features, bg, dim=1, eps=1e-6)
        if not self.contrast_head:
            contrast = torch.zeros_like(contrast)
        weighted = features * self.channel_weights(features, fg, bg)
        residual = self.head(torch.cat((weighted, contrast[:, None], logits.float()), dim=1))
        return logits + residual.to(dtype=logits.dtype)
