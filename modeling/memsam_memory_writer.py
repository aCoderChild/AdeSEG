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
    memory-encoder input location. SAM2.1 applies an affine transform to those
    probabilities before calling the memory encoder; this module reverses that
    configured transform for its cue. ``generic`` uses the same feature path
    and parameters with neutral 0.5 occupancy. The native SAM2 memory encoder
    still receives its original transformed mask unchanged in both modes.
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
        self.original_methods = {}
        self.current_frame = None
        self.mask_scale = float(getattr(model, "sigmoid_scale_for_mem_enc", 1.0))
        self.mask_bias = float(getattr(model, "sigmoid_bias_for_mem_enc", 0.0))
        if self.mask_scale == 0:
            raise ValueError("SAM2 memory-mask scale must be nonzero")

    def _pre_hook(self, module, args, kwargs):
        if args:
            input_features = args[0]
        elif "pix_feat" in kwargs:
            input_features = kwargs["pix_feat"]
        else:
            raise RuntimeError("Could not identify SAM2 memory feature input")
        if len(args) > 1:
            memory_mask = args[1]
        elif "masks" in kwargs:
            memory_mask = kwargs["masks"]
        else:
            raise RuntimeError("Could not identify SAM2 memory mask input")
        # SAM2.1 applies sigmoid_scale_for_mem_enc and sigmoid_bias_for_mem_enc
        # after sigmoid, then calls MemoryEncoder(skip_mask_sigmoid=True).
        # Reverse that exact model-configured affine transform to recover the
        # soft probability cue. Preserve `memory_mask` itself unchanged below.
        probabilities = (memory_mask.float() - self.mask_bias) / self.mask_scale
        tolerance = 8 * torch.finfo(probabilities.dtype).eps
        if (probabilities < -tolerance).any() or (probabilities > 1 + tolerance).any():
            raise RuntimeError(
                "SAM2 memory mask is inconsistent with its configured sigmoid affine transform"
            )
        probabilities = probabilities.clamp(0.0, 1.0)
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
            "mask_shape": list(memory_mask.shape),
            "mode": self.mode,
            "input_feature_sha256": _tensor_sha256(input_features),
            "memory_encoder_mask_sha256": _tensor_sha256(memory_mask),
            "recovered_probability_sha256": _tensor_sha256(probabilities),
            "memory_mask_scale": self.mask_scale,
            "memory_mask_bias": self.mask_bias,
            "adapted_feature_sha256": _tensor_sha256(updated),
        })
        return args, kwargs

    def install(self):
        if self.handle is not None:
            raise RuntimeError("Memory writer hook is already installed")
        if not hasattr(self.model, "memory_encoder"):
            raise TypeError("Model has no SAM2 memory_encoder")
        methods = {
            "track_step": lambda args, kwargs: kwargs.get("frame_idx", args[0] if args else None),
            "_run_memory_encoder": lambda args, kwargs: kwargs.get("frame_idx", args[1] if len(args) > 1 else None),
        }
        for name, get_frame in methods.items():
            if not hasattr(self.model, name):
                raise TypeError(f"Model has no {name} method for memory-write frame alignment")
            original = getattr(self.model, name)
            self.original_methods[name] = original

            def wrap(method, frame_getter, method_name):
                @functools.wraps(method)
                def aligned(*args, **kwargs):
                    frame = frame_getter(args, kwargs)
                    if frame is None:
                        raise RuntimeError(f"Could not identify SAM2 {method_name} frame index")
                    previous_frame = self.current_frame
                    self.current_frame = int(frame)
                    try:
                        return method(*args, **kwargs)
                    finally:
                        self.current_frame = previous_frame
                return aligned

            setattr(self.model, name, wrap(original, get_frame, name))
        self.handle = self.model.memory_encoder.register_forward_pre_hook(
            self._pre_hook, with_kwargs=True
        )
        return self

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        for name, method in self.original_methods.items():
            setattr(self.model, name, method)
        self.original_methods.clear()
        self.current_frame = None

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
