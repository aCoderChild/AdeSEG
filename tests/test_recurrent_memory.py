"""Guards for the MedSAM2 recurrent-memory plumbing shared by the RKN and RDE-VOS updates."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "external" / "MedSAM2")]

from modeling.kalman_memory import RKNMemoryUpdate  # noqa: E402
from modeling.rde_memory import RDEMemoryUpdate  # noqa: E402
from modeling.recurrent_memory import load_memory_update, save_memory_update  # noqa: E402


def test_checkpoints_round_trip_for_both_updates(tmp_path):
    model = SimpleNamespace(mem_dim=8, hidden_dim=16)
    for update in (RKNMemoryUpdate(8, 16, initial_variance=0.5),
                   RDEMemoryUpdate(8, 16, mode="gt-compress", mem_every=2)):
        save_memory_update(update, tmp_path / "update.pt")
        loaded = load_memory_update(tmp_path / "update.pt", model, "cpu")
        assert type(loaded) is type(update) and loaded.config == update.config
        for key, value in update.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value)
