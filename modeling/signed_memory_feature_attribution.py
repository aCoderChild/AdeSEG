"""Read-only Captum attribution of spatial memory content, not token deletion.

Signed feature attribution is sensitivity to scaling content toward zero while
positions and attention entries remain. It is neither a region label nor Dice
utility. The caller must separately verify full native trajectory provenance.
"""
import hashlib
from pathlib import Path
import sys

import torch

from modeling.token_diagnostic_replay import detached_copy


CAPTUM_SOURCE_SHA256 = "7964eedcd173cfbf10c9107b367e6de4f759ecbbe48af9cafff4aec093f5cb4c"


def load_input_x_gradient(root):
    root = Path(root).resolve()
    source = root / "captum/attr/_core/input_x_gradient.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != CAPTUM_SOURCE_SHA256:
        raise RuntimeError("Captum InputXGradient source differs from pin")
    if "captum" in sys.modules:
        if Path(sys.modules["captum"].__file__).resolve().parent != root / "captum":
            raise RuntimeError("A different Captum installation was already imported")
    else:
        sys.path.insert(0, str(root))
    from captum.attr import InputXGradient
    return InputXGradient


class SpatialMemoryScalar:
    """Replay raw native decoder scalars from [B,spatial_tokens,C] features."""

    def __init__(self, model, capture):
        if model.training or capture.layers != [0, 1, 2, 3]:
            raise ValueError("Expected an evaluation model and four-layer native capture")
        self.model = model
        self.memory = detached_copy(capture.memory)
        self.decoder = detached_copy(capture.decoder)
        self.outputs = detached_copy(capture.outputs)
        memory = self.memory["memory"]
        if memory.ndim != 3 or memory.shape[1] != 1:
            raise ValueError("This diagnostic requires one persistent object per read")
        pointers = self.memory["num_obj_ptr_tokens"]
        if not 0 <= pointers < memory.shape[0]:
            raise ValueError("Invalid spatial/pointer memory layout")
        self.spatial_count = memory.shape[0] - pointers
        self.inputs = memory[:self.spatial_count].permute(1, 0, 2).clone()
        self.pointer_tail = memory[self.spatial_count:].clone()
        self.chosen = int(self.outputs[1].argmax(dim=1).item())
        self.support = self.outputs[0][0, self.chosen] > 0

    def raw_outputs(self, spatial):
        if spatial.shape != self.inputs.shape:
            raise ValueError("Spatial memory batch/token/channel dimensions changed")
        kwargs = dict(self.memory)
        kwargs["memory"] = torch.cat((spatial.permute(1, 0, 2), self.pointer_tail), dim=0)
        conditioned = self.model.memory_attention(**kwargs)
        decoder = dict(self.decoder)
        decoder["image_embeddings"] = conditioned.permute(1, 2, 0).reshape_as(
            decoder["image_embeddings"]
        )
        return self.model.sam_mask_decoder(**decoder)

    def __call__(self, spatial, target):
        if target not in ("object_presence", "native_mask_support"):
            raise ValueError("Unknown attribution scalar")
        if target == "native_mask_support" and not self.support.any():
            raise ValueError("Native mask support is empty; target unavailable")
        masks, _, _, score = self.raw_outputs(spatial)
        if target == "object_presence":
            return score.reshape(1)
        return masks[0, self.chosen][self.support].mean().reshape(1)

    def attribute(self, captum_root, target):
        InputXGradient = load_input_x_gradient(captum_root)
        with torch.inference_mode(False), torch.enable_grad():
            inputs = self.inputs.detach().clone().requires_grad_(True)
            result = InputXGradient(lambda x: self(x, target)).attribute(inputs)
        if not torch.isfinite(result).all():
            raise RuntimeError("Nonfinite spatial-memory feature attribution")
        return result.detach()
