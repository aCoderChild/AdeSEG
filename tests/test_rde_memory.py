"""Guards for the RDE-VOS memory update."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MedSAM2")]

from modeling.rde_memory import RDEMemoryUpdate  # noqa: E402


def test_rde_starts_at_the_ema_and_learns_through_its_branches():
    torch.manual_seed(0)
    update = RDEMemoryUpdate(memory_channels=8, image_channels=16, initial_gain=0.8)
    mean, candidate = torch.randn(1, 8, 12, 12), torch.randn(1, 8, 12, 12)
    variance = update.initial_variance(mean)
    out = update.update(mean, update.predict(variance), candidate, None, None, torch.ones(1))
    assert torch.allclose(out["mean"], mean + 0.8 * (candidate - mean), atol=1e-2)
    assert torch.equal(update.readout(mean, variance), mean)

    out["mean"].pow(2).sum().backward()
    assert update.aggregate.non_local.W.weight.grad.abs().sum() > 0
    assert update.aggregate.aspp.conv1.weight.grad.abs().sum() > 0
    assert update.squeeze.weight.grad.abs().sum() > 0
