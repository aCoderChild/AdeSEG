"""Recurrent Dynamic Embedding (RDE-VOS, Li et al., CVPR 2022) update for frozen MedSAM2.

Same interface as the Kalman update and the EMA baseline, so all three share the
anchor slot, the single recurrent state S and the read path; only the update
rule differs. The update follows the official MemCrompress key branch
(github.com/Limingxing00/RDE-VOS-CVPR2022, model/modules.py):

    S_t = Squeeze(SAM(Cat_t[S_{t-1}, C_t])),  SAM(x) = NL(x) + ASPP3D(NL(x)),

with a 3D non-local block (spatial max-pool sub-sampling, no BN), a 3D ASPP
(dilations 1, 2, 4, 6) and a 2x3x3 convolution that squeezes the two time steps
into one. MedSAM2 stores one 64-channel memory map per frame, so the update acts
on that map. Deviation from the official code: it is initialised at the EMA
(S + k(C - S), k = 0.835, the Kalman memory's mean present-frame gain) with the
non-local and ASPP outputs near zero, so training learns departures from the EMA
instead of starting from a random memory.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

CHECKPOINT_KIND = "rde"


class NonLocalBlock3D(nn.Module):
    """Embedded-Gaussian-free non-local block of the official code (dot product / N, residual)."""

    def __init__(self, channels: int):
        super().__init__()
        inter = channels // 2
        self.inter = inter
        self.theta = nn.Conv3d(channels, inter, 1)
        self.phi = nn.Sequential(nn.Conv3d(channels, inter, 1), nn.MaxPool3d((1, 2, 2)))
        self.g = nn.Sequential(nn.Conv3d(channels, inter, 1), nn.MaxPool3d((1, 2, 2)))
        self.W = nn.Conv3d(inter, channels, 1)

    def forward(self, x):
        batch = x.size(0)
        g = self.g(x).view(batch, self.inter, -1).permute(0, 2, 1)
        theta = self.theta(x).view(batch, self.inter, -1).permute(0, 2, 1)
        phi = self.phi(x).view(batch, self.inter, -1)
        affinity = theta @ phi
        y = (affinity / affinity.size(-1)) @ g
        y = y.permute(0, 2, 1).reshape(batch, self.inter, *x.shape[2:])
        return self.W(y) + x


class ASPP3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = channels // reduction
        self.branches = nn.ModuleList([nn.Conv3d(channels, mid, 1, bias=False)] + [
            nn.Conv3d(channels, mid, (1, 3, 3), padding=(0, d, d), dilation=(1, d, d), bias=False)
            for d in (2, 4, 6)
        ])
        self.conv1 = nn.Conv3d(mid * 4, channels, (1, 3, 3), padding=(0, 1, 1), bias=False)

    def forward(self, x):
        return F.relu(self.conv1(torch.cat([F.relu(branch(x)) for branch in self.branches], dim=1)))


class SpatioTemporalAggregation(nn.Module):
    """SAM of the official code with repeat = 0."""

    def __init__(self, channels: int):
        super().__init__()
        self.non_local = NonLocalBlock3D(channels)
        self.aspp = ASPP3D(channels)

    def forward(self, x):
        x = self.non_local(x)
        return self.aspp(x) + x


class RDEMemoryUpdate(nn.Module):
    def __init__(self, memory_channels: int, image_channels: int, initial_gain: float = 0.835):
        super().__init__()
        self.memory_channels = memory_channels
        self.image_channels = image_channels
        self.config = {"initial_gain": initial_gain}
        self.aggregate = SpatioTemporalAggregation(memory_channels)
        self.squeeze = nn.Conv3d(memory_channels, memory_channels, (2, 3, 3), padding=(0, 1, 1))
        with torch.no_grad():
            # Start at the EMA: centre taps blend the previous state and the new memory.
            self.squeeze.weight.zero_()
            self.squeeze.bias.zero_()
            channels = torch.arange(memory_channels)
            self.squeeze.weight[channels, channels, 0, 1, 1] = 1.0 - initial_gain
            self.squeeze.weight[channels, channels, 1, 1, 1] = initial_gain
            # Near-zero residual branches (small, not zero, so the ReLUs still pass gradients).
            nn.init.normal_(self.aggregate.non_local.W.weight, std=1e-3)
            nn.init.zeros_(self.aggregate.non_local.W.bias)
            nn.init.normal_(self.aggregate.aspp.conv1.weight, std=1e-3)

    def project_image(self, image_feature):
        return None

    def initial_variance(self, mean: torch.Tensor) -> torch.Tensor:
        return mean.new_ones(mean.size(0), 1, *mean.shape[-2:])  # unused; kept for the shared interface

    def readout(self, mean, variance):
        return mean

    def predict(self, variance):
        return variance

    def update(self, mean, prior_variance, candidate, image_projection, mask_probability, presence):
        stacked = torch.stack([mean, candidate], dim=2)  # [B, C, 2, H, W]: previous RDE, newest frame
        new_mean = self.squeeze(self.aggregate(stacked)).squeeze(2)
        undefined = torch.full_like(prior_variance, float("nan"))  # no gain or noise in RDE
        return {"mean": new_mean, "variance": prior_variance, "gain": undefined, "observation_noise": undefined}
