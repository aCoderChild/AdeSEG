"""RGB-only replacement for the existing refiner's contrast head input.

The other feature/attention paths and allocated weights are unchanged. RGB is
the decoded source frame in [0,1], already cropped to the model's viewport.
No labels or inference-time thresholds are used in the contrast calculation.
"""
from pathlib import Path
import threading

import torch
from torch.nn import functional as F

from modeling.foreground_conditioned_refiner import _source_sha256
import modeling.multiscale_channel_value_refiner_v1 as baseline


def rgb_contrast(rgb, logits, attention_size):
    if rgb.ndim != 4 or rgb.shape[1] != 3 or logits.ndim != 4 or logits.shape[1] != 1 \
            or rgb.shape[0] != logits.shape[0] or rgb.device != logits.device:
        raise ValueError('RGB and logit geometry or device differs')
    if not torch.isfinite(rgb).all() or not torch.isfinite(logits).all() \
            or (rgb < 0).any() or (rgb > 1).any():
        raise ValueError('RGB or logits are invalid')
    size = logits.shape[-2:]
    coarse = (min(size[0], attention_size), min(size[1], attention_size))
    colour = F.interpolate(rgb.float(), size=coarse, mode='area')
    probability = F.interpolate(torch.sigmoid(logits.float()), size=coarse, mode='area')
    foreground = (colour * probability).sum((-2, -1), keepdim=True) / probability.sum(
        (-2, -1), keepdim=True).clamp_min(1e-6)
    background_weight = 1 - probability
    background = (colour * background_weight).sum((-2, -1), keepdim=True) / background_weight.sum(
        (-2, -1), keepdim=True).clamp_min(1e-6)
    contrast = F.cosine_similarity(colour, foreground, dim=1, eps=1e-6) - \
               F.cosine_similarity(colour, background, dim=1, eps=1e-6)
    return F.interpolate(contrast[:, None], size=size, mode='bilinear', align_corners=False)


class RGBContrastControlV1(baseline.MultiscaleChannelValueRefinerV1):
    def __init__(self, **kwargs):
        if 'mode' in kwargs:
            raise ValueError('RGB control has a fixed baseline mode')
        super().__init__(mode='foreground_background_asinh', **kwargs)
        self._rgb_forward_lock = threading.Lock()

    def forward(self, early, deep, logits, rgb):
        if not self._rgb_forward_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent or reentrant RGB control call')
        handle = None
        try:
            contrast = rgb_contrast(rgb, logits, self.attention_size)
            calls = []

            def replace(_module, args):
                x = args[0]
                if len(args) != 1 or x.ndim != 4 or x.shape[1] != self.hidden + 2 \
                        or x.shape[-2:] != contrast.shape[-2:]:
                    raise ValueError('Unexpected refiner head input')
                calls.append(1)
                return (torch.cat((x[:, :-2], contrast.to(x.dtype), x[:, -1:]), dim=1),)

            handle = self.head.register_forward_pre_hook(replace)
            result = super().forward(early, deep, logits)
            if len(calls) != 1:
                raise ValueError('Refiner head invocation differs')
            return result
        finally:
            if handle is not None:
                handle.remove()
            self._rgb_forward_lock.release()

    def load_trained_state(self, payload, **provenance):
        self._trained_checkpoint_loaded = False
        if not isinstance(payload, dict) or payload.get('format') != 'rgb_contrast_control_v1' \
                or payload.get('control_source_sha256') != _source_sha256(Path(__file__)) \
                or payload.get('baseline_source_sha256') != _source_sha256(Path(baseline.__file__)):
            raise ValueError('RGB control checkpoint or source differs')
        compatible = dict(payload, format='multiscale_channel_value_refiner_v1',
                          source_sha256=payload['baseline_source_sha256'],
                          mode='foreground_background_asinh')
        super().load_trained_state(compatible, **provenance)
