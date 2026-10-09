"""RDE-VOS-derived key-compression control using official components from external/RDE-VOS-CVPR2022."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / "external" / "RDE-VOS-CVPR2022"
if str(OFFICIAL_ROOT) not in sys.path:
    sys.path.append(str(OFFICIAL_ROOT))  # the official code imports its packages as `model` and `util`

from model.modules import SAM  # noqa: E402

MODES = ("gt-compress", "two-frames-compress")

# Parameter names of the earlier re-implementation (EMA initialised, gt-compress, every frame).
_LEGACY_NAMES = {
    "aggregate.non_local.": "key_encoder.non_local.",
    "aggregate.aspp.branches.0.": "key_encoder.conv1.aspp1.atrous_conv.",
    "aggregate.aspp.branches.1.": "key_encoder.conv1.aspp2.atrous_conv.",
    "aggregate.aspp.branches.2.": "key_encoder.conv1.aspp3.atrous_conv.",
    "aggregate.aspp.branches.3.": "key_encoder.conv1.aspp4.atrous_conv.",
    "aggregate.aspp.conv1.": "key_encoder.conv1.conv1.",
    "squeeze.": "compress_key.",
}


class RDEMemoryUpdate(nn.Module):
    CHECKPOINT_KIND = "rde"

    def __init__(self, memory_channels: int, image_channels: int, mode: str = "two-frames-compress",
                 mem_every: int = 3, repeat: int = 0):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}.")
        self.memory_channels = memory_channels
        self.image_channels = image_channels
        self.mode = mode
        self.mem_every = mem_every
        self.config = {"mode": mode, "mem_every": mem_every, "repeat": repeat}
        # MemCrompress.key_encoder / compress_key, with the official names and initialisation.
        self.key_encoder = SAM(memory_channels, memory_channels, repeat=repeat)
        self.compress_key = nn.Conv3d(memory_channels, memory_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1))

    @property
    def keeps_recent(self) -> bool:
        return self.mode == "two-frames-compress"

    @staticmethod
    def upgrade(config: dict, state_dict: dict) -> tuple[dict, dict]:
        """Checkpoint of the earlier re-implementation -> this module (same computation)."""
        if "initial_gain" not in config:
            return config, state_dict
        renamed = {}
        for key, value in state_dict.items():
            for old, new in _LEGACY_NAMES.items():
                if key.startswith(old):
                    key = new + key[len(old):]
                    break
            renamed[key] = value
        return {"mode": "gt-compress", "mem_every": 1, "repeat": 0}, renamed

    def updates_at(self, frame_idx: int) -> bool:
        """Official InferenceCore.do_pass: a frame is added to memory when ti % mem_every == 0."""
        return frame_idx % self.mem_every == 0

    def memory_slots(self, state, frame_idx, num_maskmem):
        """(memory, frames ago) read at ``frame_idx``, as MemoryBank.match_memory: before the
        first compression the RDE is the prompt memory itself."""
        oldest = num_maskmem - 1
        if self.mode == "gt-compress" or state.recent is None:
            return [(state.anchor, 0), (state.mean, oldest)]
        return [(state.anchor, 0), (state.anchor, 0), (state.mean, oldest),
                (state.recent, frame_idx - state.recent_frame_idx)]

    def start(self, memory: torch.Tensor) -> None:
        return None  # the RDE keeps no covariance

    def update(self, state, covariance, memory, reliability=None):
        """MemoryBank.add_memory (compress modes): RDE_t = compress_key(SAM(cat_t[RDE_t-1, k_t]))."""
        stacked = torch.stack([state, memory], dim=2)  # [B, C, 2, H, W]
        return {"mean": self.compress_key(self.key_encoder(stacked)).squeeze(2), "covariance": None}
