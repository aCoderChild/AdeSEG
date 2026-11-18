"""Scientific blocks preserve source identity and training/inference boundaries."""
import torch

from modeling.blocks.teacher_preserving_kl import (
    TeacherPreservingKLLoss,
    bernoulli_teacher_kl,
)
from modeling.blocks.medsam2_backbone import MEDSAM2_LATEST_SHA256
from modeling.blocks.high_resolution_adapter import MedicalHighResolutionAdapter
from modeling.blocks.multiscale_feature_aggregation import MultistageFeatureAggregator
from modeling.blocks.lean_sam2_backbone import indexed_spatial_rope_mps
from modeling.feature_aggregation import MultistageFeatureAggregator as LegacyPublicLean
from modeling.medical_highres.adapter import MedicalHighResolutionAdapter as OriginalMedical
from modeling.medical_highres.teacher_loss import bernoulli_teacher_kl as OriginalKL
from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1


def test_blocks_are_compatibility_preserving_aliases():
    assert MultistageFeatureAggregator is LegacyPublicLean
    assert issubclass(MultistageFeatureAggregator, MultiscaleChannelValueRefinerV1)
    assert MedicalHighResolutionAdapter is OriginalMedical
    assert bernoulli_teacher_kl is OriginalKL
    assert callable(indexed_spatial_rope_mps)
    assert len(MEDSAM2_LATEST_SHA256) == 64


def test_teacher_kl_block_detaches_native_teacher():
    student = torch.randn(1, 1, 3, 4, requires_grad=True)
    teacher = torch.randn(1, 1, 3, 4, requires_grad=True)
    loss = TeacherPreservingKLLoss(temperature=1.0)(student, teacher)
    assert torch.allclose(loss, OriginalKL(student, teacher))
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_teacher_kl_rejects_invalid_temperature():
    try:
        TeacherPreservingKLLoss(temperature=0)
    except ValueError:
        pass
    else:
        raise AssertionError("KL temperature cannot be zero")
