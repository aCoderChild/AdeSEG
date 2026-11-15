"""Matched fixed-read controls for the region-conditioned research refiner.

This module reuses the same learned projections, author channel gate and head;
it changes only which input information is available. It is not authorized for
live video inference. The fixed spatial permutation is label independent.
"""
import torch
import torch.nn.functional as F

from modeling.foreground_conditioned_refiner import ForegroundConditionedRefiner


ARMS = ('early_only', 'deep_only', 'foreground_only', 'rgb_contrast', 'spatial_shuffle')
SHUFFLE_SEED = 20261115


def spatial_shuffle(features):
    """Apply one fixed, shared, bijective permutation to aligned feature pixels."""
    if features.ndim != 4:
        raise ValueError('Expected NCHW features')
    h, w = features.shape[-2:]
    generator = torch.Generator(device='cpu').manual_seed(SHUFFLE_SEED)
    order = torch.randperm(h * w, generator=generator, device='cpu').to(features.device)
    return features.flatten(2).index_select(2, order).reshape_as(features)


class SourceGroupAblationRefiner(ForegroundConditionedRefiner):
    def __init__(self, *, control):
        if control not in (*ARMS, 'reference'):
            raise ValueError('Unknown fixed-read control')
        super().__init__(gate_mode='region', contrast_head=True)
        self.control = control

    def load_trained_state(self, payload, **kwargs):
        self._trained_checkpoint_loaded = False
        if not isinstance(payload, dict) or payload.get('control') != self.control:
            raise ValueError('Checkpoint control differs')
        super().load_trained_state(payload, **kwargs)

    def forward(self, early, deep, logits, rgb=None):
        if (not self.training or not torch.is_grad_enabled()) and not self._trained_checkpoint_loaded:
            raise RuntimeError('Untrained refinement cannot be used in evaluation')
        if (early.ndim != 4 or deep.ndim != 4 or logits.ndim != 4 or logits.shape[1] != 1
                or early.shape[1] != self.early_channels or deep.shape[1] != self.deep_channels
                or early.shape[0] != deep.shape[0] or early.shape[0] != logits.shape[0]):
            raise ValueError('Feature or mask geometry differs')
        if early.device != deep.device or early.device != logits.device or not all(
                torch.isfinite(t).all() for t in (early, deep, logits)):
            raise ValueError('Inputs must be finite and on one device')
        if self.control == 'rgb_contrast':
            if (rgb is None or rgb.ndim != 4 or rgb.shape[:2] != (logits.shape[0], 3)
                    or rgb.device != logits.device or not torch.isfinite(rgb).all()):
                raise ValueError('RGB control requires finite, aligned NCHW RGB')
        elif rgb is not None:
            raise ValueError('Unexpected RGB input for feature control')
        if self.control == 'early_only':
            deep = torch.zeros_like(deep)
        elif self.control == 'deep_only':
            early = torch.zeros_like(early)
        size = logits.shape[-2:]
        e = F.interpolate(self.early(early.float()), size=size, mode='bilinear', align_corners=False)
        d = F.interpolate(self.deep(deep.float()), size=size, mode='bilinear', align_corners=False)
        features = torch.cat((e, d), dim=1)
        if self.control == 'spatial_shuffle':
            features = spatial_shuffle(features)
        prob = torch.sigmoid(logits.float())
        fg = (features * prob).sum((-2, -1), keepdim=True) / prob.sum((-2, -1), keepdim=True).clamp_min(1e-6)
        bg = (features * (1-prob)).sum((-2, -1), keepdim=True) / (1-prob).sum((-2, -1), keepdim=True).clamp_min(1e-6)
        if self.control == 'foreground_only':
            contrast = F.cosine_similarity(features, fg, dim=1, eps=1e-6)
            weights = self.channel_weights(features, fg, torch.zeros_like(bg))
        elif self.control == 'rgb_contrast':
            color = F.interpolate(rgb.float(), size=size, mode='bilinear', align_corners=False)
            c_fg = (color * prob).sum((-2, -1), keepdim=True) / prob.sum((-2, -1), keepdim=True).clamp_min(1e-6)
            c_bg = (color * (1-prob)).sum((-2, -1), keepdim=True) / (1-prob).sum((-2, -1), keepdim=True).clamp_min(1e-6)
            contrast = F.cosine_similarity(color, c_fg, dim=1, eps=1e-6) - F.cosine_similarity(color, c_bg, dim=1, eps=1e-6)
            weights = self.channel_weights(features, fg, bg)
        else:
            contrast = F.cosine_similarity(features, fg, dim=1, eps=1e-6) - F.cosine_similarity(features, bg, dim=1, eps=1e-6)
            weights = self.channel_weights(features, fg, bg)
        residual = self.head(torch.cat((features * weights, contrast[:, None], logits.float()), dim=1))
        return logits + residual.to(dtype=logits.dtype)
