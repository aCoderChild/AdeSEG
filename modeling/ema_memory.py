"""Constant-gain (EMA) memory baseline for frozen MedSAM2.

This is the ablation the Kalman memory is compared against: the same single
recursive spatial state and the same memory read path, but the adaptive Kalman
gain ``K = P⁻ / (P⁻ + R)`` is replaced by a fixed present/absent gain. If the
Kalman filter cannot beat this, its benefit is the recursive 1-slot structure,
not the adaptive gain (see ``docs/EXPERIMENTS.md``).

It implements the same interface ``KalmanMemoryMixin`` expects of a memory
update module (``project_image``, ``initial_variance``, ``readout``, ``predict``,
``update``), so it drops into the predictor in place of ``KalmanMemoryUpdate``
with no change to the mixin. The read path matches the untrained Kalman module
(zero uncertainty embedding), so the two differ only in the write gain.
"""

from __future__ import annotations

import torch


class ConstantGainMemory:
    """EMA memory: one recursive state written with a fixed present/absent gain."""

    def __init__(
        self,
        memory_channels: int,
        image_channels: int,
        present_gain: float,
        absent_gain: float,
        initial_variance: float = 1.0,
        process_noise: float = 1.0,
        min_variance: float = 1e-4,
        max_variance: float = 1e4,
    ):
        self.memory_channels = memory_channels
        self.image_channels = image_channels
        self.present_gain = present_gain
        self.absent_gain = absent_gain
        self.initial_variance_value = initial_variance
        self.process_noise = process_noise
        self.min_variance = min_variance
        self.max_variance = max_variance

    def project_image(self, image_feature: torch.Tensor):
        """Unused: EMA needs no image projection (it computes no measurement noise)."""
        return None

    def initial_variance(self, mean: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (mean.size(0), 1, *mean.shape[-2:]), self.initial_variance_value,
            dtype=mean.dtype, device=mean.device,
        )

    def readout(self, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        return mean  # matches the untrained Kalman read path (uncertainty embedding = 0)

    def predict(self, variance: torch.Tensor, noise_scale=None) -> torch.Tensor:
        return (variance + self.process_noise).clamp(self.min_variance, self.max_variance)

    def update(
        self, mean, prior_variance, candidate, image_projection, mask_probability, presence,
        reliability=None, noise_scale=None,
    ):
        constant = self.present_gain if bool((presence > 0.5).all()) else self.absent_gain
        gain = torch.full_like(prior_variance, constant)
        return {
            "mean": mean + gain * (candidate - mean),
            "variance": ((1.0 - gain) * prior_variance).clamp(self.min_variance, self.max_variance),
            "gain": gain,
            "observation_noise": torch.zeros_like(prior_variance),
        }
