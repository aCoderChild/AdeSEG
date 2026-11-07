"""Soft-mask-conditioned residual on the feature entering SAM2 memory.

The local convolutions come from MemSAM's public ForegroundReinforcingModule.
This SAM2 adaptation uses that learned feature path as an additive residual,
initialized to exact identity. It deliberately does not copy MemSAM's final
mask-times-softmax output, which would suppress contextual features.
"""

from __future__ import annotations

import hashlib
import importlib
import functools
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


MEMSAM_REVISION = "36f1b988f14b518a6346d2edb1267340387e646c"
MEMSAM_SOURCE_SHA256 = "71a959a61f544da2ca945bdfccb880d66e5e355da4c48f351230422a59b70dd9"


def _author_module_class():
    root = Path(__file__).resolve().parents[1] / "external" / "MemSAM"
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    source = root / "models/segment_anything_memsam/modeling/mem_modules.py"
    if revision != MEMSAM_REVISION or status:
        raise RuntimeError("MemSAM must be the pinned clean upstream checkout")
    if hashlib.sha256(source.read_bytes()).hexdigest() != MEMSAM_SOURCE_SHA256:
        raise RuntimeError("MemSAM reinforcement module source does not match its pin")

    # Load only the upstream modeling module. Its package __init__ imports the
    # complete old SAMUS stack, including optional dependencies not needed here.
    package_root = root / "models/segment_anything_memsam"
    for name, path in (
        ("_adeseg_memsam", package_root),
        ("_adeseg_memsam.modeling", package_root / "modeling"),
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    module = importlib.import_module("_adeseg_memsam.modeling.mem_modules")
    return module.ForegroundReinforcingModule


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class SoftMaskResidualWriter(nn.Module):
    """Apply a zero-initialized, continuous mask-conditioned memory correction.

    ``regional`` uses area-averaged soft foreground probabilities at each
    memory-encoder input location. ``generic`` uses the same feature path and
    parameters with neutral 0.5 occupancy. The native SAM2 memory encoder still
    receives its original mask unchanged in both modes.
    """

    def __init__(self, channels: int = 256, mid_channels: int = 128, kernel: int = 3):
        super().__init__()
        if channels < 1 or mid_channels < 1 or kernel < 1 or kernel % 2 == 0:
            raise ValueError("Expected positive channels and an odd convolution kernel")
        author_module = _author_module_class()
        self.author_path = author_module(channels, mid_channels, channels, kernel)
        # Exact SAM2 identity at initialization; the final author convolution
        # learns the residual after receiving a nonzero gradient.
        nn.init.zeros_(self.author_path.conv_1x1.weight)
        if self.author_path.conv_1x1.bias is not None:
            nn.init.zeros_(self.author_path.conv_1x1.bias)
        self.channels = channels

    def forward(self, features: torch.Tensor, probabilities: torch.Tensor, mode: str):
        if mode not in ("regional", "generic"):
            raise ValueError("mode must be 'regional' or 'generic'")
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError("Expected SAM2 memory input [B,C,H,W]")
        if probabilities.ndim != 4 or probabilities.shape[:2] != (features.shape[0], 1):
            raise ValueError("Expected memory mask probabilities [B,1,H,W]")
        if not torch.isfinite(features).all() or not torch.isfinite(probabilities).all():
            raise ValueError("Memory inputs must be finite")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("Expected probabilities in [0,1]")

        branch_dtype = self.author_path.conv_wxh.weight.dtype
        branch_features = features.to(dtype=branch_dtype)
        probability = probabilities.to(device=features.device, dtype=branch_dtype)
        probability = F.interpolate(
            probability, size=features.shape[-2:], mode="area"
        )
        if mode == "generic":
            probability = torch.full_like(probability, 0.5)

        # Reuse MemSAM's published Conv3x3 -> Conv1x1 feature path. The
        # projection is additive and has no token/frame keep-drop operation.
        joined = torch.cat((probability, branch_features), dim=1)
        hidden = self.author_path.conv_wxh(joined)
        delta = self.author_path.conv_1x1(hidden)
        return features + delta.to(dtype=features.dtype)


class NativeMemoryWriter:
    """Install a pre-hook at SAM2's memory-encoder input without editing SAM2.

    The hook changes only ``pix_feat``. The original mask, encoder parameters,
    positional encoding, memory attention, pointers, and decoder are untouched.
    Call ``close`` or use this object as a context manager to remove the hook.
    """

    def __init__(self, model, writer: SoftMaskResidualWriter, mode: str = "regional"):
        if mode not in ("regional", "generic"):
            raise ValueError("mode must be 'regional' or 'generic'")
        self.model = model
        self.writer = writer
        self.mode = mode
        self.handle = None
        self.records = []
        self.original_track_step = None
        self.current_frame = None

    def _pre_hook(self, module, args, kwargs):
        if args:
            input_features = args[0]
        elif "pix_feat" in kwargs:
            input_features = kwargs["pix_feat"]
        else:
            raise RuntimeError("Could not identify SAM2 memory feature input")
        if len(args) > 1:
            probabilities = args[1]
        elif "masks" in kwargs:
            probabilities = kwargs["masks"]
        else:
            raise RuntimeError("Could not identify SAM2 memory mask input")
        updated = self.writer(input_features, probabilities, self.mode)
        if args:
            args = (updated, *args[1:])
        else:
            kwargs = dict(kwargs)
            kwargs["pix_feat"] = updated
        if self.current_frame is None:
            raise RuntimeError("Memory write occurred outside a tracked frame step")
        self.records.append({
            "frame": self.current_frame,
            "feature_shape": list(input_features.shape),
            "mask_shape": list(probabilities.shape),
            "mode": self.mode,
            "input_feature_sha256": _tensor_sha256(input_features),
            "mask_input_sha256": _tensor_sha256(probabilities),
            "adapted_feature_sha256": _tensor_sha256(updated),
        })
        return args, kwargs

    def install(self):
        if self.handle is not None:
            raise RuntimeError("Memory writer hook is already installed")
        if not hasattr(self.model, "memory_encoder"):
            raise TypeError("Model has no SAM2 memory_encoder")
        if not hasattr(self.model, "track_step"):
            raise TypeError("Model has no track_step method for memory-write frame alignment")
        self.original_track_step = self.model.track_step

        @functools.wraps(self.original_track_step)
        def track_step_with_frame(*args, **kwargs):
            frame = kwargs.get("frame_idx")
            if frame is None and args:
                frame = args[0]
            if frame is None:
                raise RuntimeError("Could not identify SAM2 track_step frame index")
            previous_frame = self.current_frame
            self.current_frame = int(frame)
            try:
                return self.original_track_step(*args, **kwargs)
            finally:
                self.current_frame = previous_frame

        self.model.track_step = track_step_with_frame
        self.handle = self.model.memory_encoder.register_forward_pre_hook(
            self._pre_hook, with_kwargs=True
        )
        return self

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.original_track_step is not None:
            self.model.track_step = self.original_track_step
            self.original_track_step = None
        self.current_frame = None

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
