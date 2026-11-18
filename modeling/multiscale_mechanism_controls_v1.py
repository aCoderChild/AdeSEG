"""Separately versioned input ablations of the asinh channel-value refiner.

The baseline implementation and its checkpoints remain untouched. These
controls allocate the same parameter tensors, but zeroed branches have fewer
effective input-dependent parameters. The spatial shuffle destroys both
within-scale geometry and cross-scale alignment; it is not a pure test of
either relationship alone. It uses a fixed permutation, not label data.
"""
from pathlib import Path

import torch

from modeling.foreground_conditioned_refiner import _source_sha256
import modeling.multiscale_channel_value_refiner_v1 as baseline


def spatial_shuffle(x, *, seed):
    """Apply one deterministic, spatially bijective permutation per tensor.

    Each spatial size has its own fixed permutation, shared across the batch
    and channels. A private CPU generator leaves training RNG streams intact.
    """
    if x.ndim != 4:
        raise ValueError('Expected a four-dimensional feature tensor')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    permutation = torch.randperm(x.shape[-2] * x.shape[-1], generator=generator)
    return x.flatten(2).index_select(2, permutation.to(x.device)).reshape_as(x)


class MultiscaleMechanismControlV1(baseline.MultiscaleChannelValueRefinerV1):
    CONTROLS = ('early_only', 'deep_only', 'spatial_shuffle')

    def __init__(self, control, **kwargs):
        if control not in self.CONTROLS or 'mode' in kwargs:
            raise ValueError('Unknown mechanism control')
        super().__init__(mode='foreground_background_asinh', **kwargs)
        self.control = control

    def forward(self, early, deep, logits):
        # Suppressing a branch must not conceal a corrupt captured tensor.
        if not all(torch.isfinite(x).all() for x in (early, deep, logits)):
            raise ValueError('Inputs must be finite')
        if self.control == 'early_only':
            deep = torch.zeros_like(deep)
        elif self.control == 'deep_only':
            early = torch.zeros_like(early)
        else:
            early = spatial_shuffle(early, seed=2107)
            deep = spatial_shuffle(deep, seed=2108)
        return super().forward(early, deep, logits)

    def load_trained_state(self, payload, **provenance):
        """Require a control-specific checkpoint as well as baseline identity."""
        self._trained_checkpoint_loaded = False
        if not isinstance(payload, dict) or payload.get('format') != 'multiscale_mechanism_controls_v1' \
                or payload.get('control') != self.control \
                or payload.get('control_source_sha256') != _source_sha256(Path(__file__)) \
                or payload.get('baseline_source_sha256') != _source_sha256(Path(baseline.__file__)):
            raise ValueError('Control checkpoint or source differs')
        # The baseline validates its full training/config/feature provenance.
        compatible = dict(payload, format='multiscale_channel_value_refiner_v1',
                          source_sha256=payload['baseline_source_sha256'],
                          mode='foreground_background_asinh')
        super().load_trained_state(compatible, **provenance)
