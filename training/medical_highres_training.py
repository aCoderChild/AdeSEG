"""Training-only frozen-MedSAM2 decoder-capture utilities.

A capture is a deterministic TRAIN sample of native decoder inputs. Frozen
decoder parameters stay unmodified; only the 5-layer medical adapter trains.
This new pipeline is NOT a reconstruction of historical multi-frame dynamics.
"""
import torch
from torch.nn import functional as F

from modeling.medical_highres import MedicalHighResolutionAdapter
from modeling.medical_highres.teacher_loss import supervised_mask_teacher_loss

REQUIRED = frozenset((
    "image_embeddings", "image_pe", "sparse_prompt_embeddings",
    "dense_prompt_embeddings", "high_res_0", "high_res_1", "target",
))


def validate_decoder_capture(capture):
    if not isinstance(capture, dict) or set(capture) != REQUIRED:
        raise ValueError(f"Decoder capture must contain exactly {sorted(REQUIRED)}")
    if any(not isinstance(x, torch.Tensor) or not torch.is_floating_point(x)
           or not x.isfinite().all().item() for x in capture.values()):
        raise ValueError("Decoder capture must contain finite float tensors")
    x, pe, s, d, h0, h1, target = (capture[k] for k in (
        "image_embeddings", "image_pe", "sparse_prompt_embeddings",
        "dense_prompt_embeddings", "high_res_0", "high_res_1", "target"))
    batch = x.shape[0] if x.ndim == 4 else 0
    if (batch != 1 or x.shape[1] != 256
            or pe.ndim != 4 or pe.shape[1:] != x.shape[1:]
            or pe.shape[0] != 1 or s.ndim != 3 or s.shape[0] != batch
            or d.shape != x.shape or h0.ndim != 4 or h0.shape[:2] != (batch, 32)
            or h1.ndim != 4 or h1.shape[:2] != (batch, 64)
            or h0.shape[-2:] != tuple(dim * 4 for dim in x.shape[-2:])
            or h1.shape[-2:] != tuple(dim * 2 for dim in x.shape[-2:])
            or target.ndim != 4 or target.shape[:2] != (batch, 1)):
        raise ValueError("Decoder capture has unsupported Hiera-T/512 geometry")
    if (target < 0).any().item() or (target > 1).any().item():
        raise ValueError("Capture target must be in [0,1]")
    return capture


def _call_decoder(decoder, capture, highres):
    return decoder(
        image_embeddings=capture["image_embeddings"],
        image_pe=capture["image_pe"],
        sparse_prompt_embeddings=capture["sparse_prompt_embeddings"],
        dense_prompt_embeddings=capture["dense_prompt_embeddings"],
        multimask_output=False, repeat_image=False,
        high_res_features=highres,
    )[0]


def medical_training_step(decoder, adapter, capture, *, teacher_weight=0.1,
                          positive_weight=0.0):
    """Differentiable adapter loss with a detached native decoder prediction."""
    if not isinstance(adapter, MedicalHighResolutionAdapter):
        raise TypeError("Expected MedicalHighResolutionAdapter")
    if decoder.training or any(p.requires_grad for p in decoder.parameters()):
        raise ValueError("Native MedSAM2 decoder must be frozen and in eval mode")
    validate_decoder_capture(capture)
    original_skips = [capture["high_res_0"], capture["high_res_1"]]
    with torch.no_grad():
        teacher = _call_decoder(decoder, capture, original_skips).detach()
    adapted_first = adapter(capture["high_res_0"],
                            capture["image_embeddings"], capture["high_res_0"])
    student = _call_decoder(
        decoder, capture, [adapted_first, capture["high_res_1"]])
    if teacher.shape != student.shape or student.shape[1] != 1:
        raise ValueError("Decoder must produce a single aligned mask logit")
    target = F.interpolate(
        capture["target"], size=student.shape[-2:], mode="nearest")
    return supervised_mask_teacher_loss(
        student, teacher, target, teacher_weight=teacher_weight,
        positive_weight=positive_weight)
