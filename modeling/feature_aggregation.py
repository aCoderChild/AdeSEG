"""Experimental trainable channel-value attention on frozen Lean reads.

This is a project adapter, not the CMSA, CFBI, or TCMN model. It keeps the
existing SAM2 output logit as its residual anchor and does not implement video
routing. Foreground/background probabilities are taken from that same logit.
The four modes have identical parameter tensors; ``diagonal`` has fewer
effective channel interactions by design. No untrained inference is allowed.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import hashlib


def _source_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def channel_matrix(query, key, probability, mode, eps=1e-6):
    """Build [B,C,C] channel attention from [B,C,H,W] Q/K and soft masks.

    Each channel's spatial profile is L2-normalized. The same soft region
    probability multiplies both Q and K, so the unnormalized inner product is
    weighted by p squared (or (1-p) squared), not p. FG/BG uses the difference
    of two soft-region correlation matrices. The diagonal control can only
    scale the corresponding value channel; all other modes mix channel values.
    Its diagonal sigmoid intentionally differs in normalization and amplitude
    from row-softmax (which would be one on a one-element diagonal), so this is
    a combined diagonal/normalization control, not a pure isolation of mixing.
    """
    if mode not in ('diagonal', 'global', 'foreground', 'foreground_background'):
        raise ValueError('Unknown channel-attention mode')
    if query.ndim != 4 or query.shape != key.shape or probability.shape != (
            query.shape[0], 1, query.shape[2], query.shape[3]):
        raise ValueError('Channel attention tensor geometry differs')
    if query.device != key.device or query.device != probability.device:
        raise ValueError('Channel attention devices differ')
    if not all(torch.isfinite(x).all() for x in (query, key, probability)):
        raise ValueError('Nonfinite channel attention input')
    if (probability < 0).any() or (probability > 1).any():
        raise ValueError('Region probabilities must be in [0,1]')

    def correlation(weight):
        q = F.normalize((query * weight).flatten(2), p=2, dim=-1, eps=eps)
        k = F.normalize((key * weight).flatten(2), p=2, dim=-1, eps=eps)
        return q @ k.transpose(1, 2)

    if mode in ('diagonal', 'global'):
        score = correlation(torch.ones_like(probability))
    elif mode == 'foreground':
        score = correlation(probability)
    else:
        score = correlation(probability) - correlation(1 - probability)
    if mode == 'diagonal':
        return torch.diag_embed(torch.sigmoid(torch.diagonal(score, dim1=1, dim2=2)))
    return score.softmax(dim=-1)


class MultiscaleChannelValueRefinerV1(nn.Module):
    """Project early/deep features, mix channel values, refine native logits.

    All four ablations have the same instantiated parameters. The deliberate
    diagonal control is not a capacity-matched dense attention mechanism: it
    has the same allocated weights but fewer effective channel interactions.
    Pooling limits the channel-correlation cost and does not use labels.
    """
    MODES = ('mask_only', 'diagonal', 'global', 'global_contrast', 'foreground',
             'foreground_background_no_contrast', 'foreground_background',
             'foreground_background_asinh')

    def __init__(self, mode='foreground_background', early_channels=32,
                 deep_channels=256, hidden=64, attention_size=32):
        super().__init__()
        if mode not in self.MODES or hidden < 2 or hidden % 2 or attention_size < 1:
            raise ValueError('Invalid channel refiner configuration')
        self.mode = mode
        self.early_channels, self.deep_channels = early_channels, deep_channels
        self.hidden, self.attention_size = hidden, attention_size
        self.early = nn.Conv2d(early_channels, hidden // 2, 1)
        self.deep = nn.Conv2d(deep_channels, hidden // 2, 1)
        self.query = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.key = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.value = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.mix_out = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.head = nn.Sequential(nn.Conv2d(hidden + 2, hidden, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self._trained_checkpoint_loaded = False

    def load_trained_state(self, payload, *, backbone_sha256, feature_pipeline_sha256,
                           train_manifest_sha256):
        """Revoke authorization on any mismatch, including partial loads."""
        self._trained_checkpoint_loaded = False
        if not isinstance(payload, dict) or payload.get('format') != 'multiscale_channel_value_refiner_v1' \
                or payload.get('research_only') is not True \
                or type(payload.get('training_steps')) is not int or payload['training_steps'] <= 0:
            raise ValueError('Matching trained research checkpoint required')
        if payload.get('source_sha256') != _source_sha256(Path(__file__)) or payload.get('mode') != self.mode:
            raise ValueError('Refiner source or ablation differs')
        config = {'early_channels': self.early_channels, 'deep_channels': self.deep_channels,
                  'hidden': self.hidden, 'attention_size': self.attention_size}
        if payload.get('config') != config:
            raise ValueError('Refiner feature or attention geometry differs')
        for name, expected in (('backbone_sha256', backbone_sha256),
                               ('feature_pipeline_sha256', feature_pipeline_sha256),
                               ('train_manifest_sha256', train_manifest_sha256)):
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected) or payload.get(name) != expected:
                raise ValueError(f'Checkpoint provenance mismatch: {name}')
        self.load_state_dict(payload['state_dict'], strict=True)
        if any(not torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError('Nonfinite checkpoint')
        self._trained_checkpoint_loaded = True

    def forward(self, early, deep, logits):
        if (not self.training or not torch.is_grad_enabled()) and not self._trained_checkpoint_loaded:
            raise RuntimeError('Untrained refinement cannot be used in evaluation')
        if early.ndim != 4 or deep.ndim != 4 or logits.ndim != 4 or logits.shape[1] != 1 \
                or early.shape[1] != self.early_channels or deep.shape[1] != self.deep_channels \
                or early.shape[0] != deep.shape[0] or early.shape[0] != logits.shape[0]:
            raise ValueError('Feature or mask geometry differs')
        if early.device != logits.device or deep.device != logits.device \
                or not all(torch.isfinite(x).all() for x in (early, deep, logits)):
            raise ValueError('Inputs must be finite and on one device')
        size = logits.shape[-2:]
        coarse = (min(size[0], self.attention_size), min(size[1], self.attention_size))
        if self.mode == 'mask_only':
            early, deep = torch.zeros_like(early), torch.zeros_like(deep)
        e = F.interpolate(self.early(early.float()), size=size, mode='bilinear', align_corners=False)
        d = F.interpolate(self.deep(deep.float()), size=size, mode='bilinear', align_corners=False)
        fine_features = torch.cat((e, d), dim=1)
        features = F.adaptive_avg_pool2d(fine_features, coarse)
        p = F.interpolate(torch.sigmoid(logits.float()), size=coarse, mode='area')
        matrix_mode = ('global' if self.mode in ('mask_only','global_contrast') else
                       'foreground_background' if self.mode in ('foreground_background_no_contrast',
                                                                 'foreground_background_asinh') else self.mode)
        matrix = channel_matrix(self.query(features), self.key(features), p, matrix_mode)
        values = self.value(features)
        mixed = (matrix @ values.flatten(2)).reshape_as(values)
        fused = fine_features + F.interpolate(self.mix_out(mixed), size=size, mode='bilinear', align_corners=False)
        fg = (features * p).sum((-2, -1), keepdim=True) / p.sum((-2, -1), keepdim=True).clamp_min(1e-6)
        bg_weight = 1 - p
        bg = (features * bg_weight).sum((-2, -1), keepdim=True) / bg_weight.sum((-2, -1), keepdim=True).clamp_min(1e-6)
        fg_similarity = F.cosine_similarity(features, fg, dim=1, eps=1e-6)
        bg_similarity = F.cosine_similarity(features, bg, dim=1, eps=1e-6)
        if self.mode in ('mask_only', 'diagonal', 'global', 'foreground_background_no_contrast'):
            contrast = torch.zeros_like(fg_similarity)
        elif self.mode == 'foreground':
            contrast = fg_similarity
        else:
            contrast = fg_similarity - bg_similarity
        contrast = F.interpolate(contrast[:, None], size=size, mode='bilinear', align_corners=False)
        head_logits=logits.float()
        if self.mode=='foreground_background_asinh':
            head_logits=torch.asinh(head_logits)
        residual = self.head(torch.cat((fused, contrast, head_logits), dim=1))
        return logits + residual.to(dtype=logits.dtype)


# Clear scientific-facing alias; the historical class name remains valid.
MultistageFeatureAggregator = MultiscaleChannelValueRefinerV1
