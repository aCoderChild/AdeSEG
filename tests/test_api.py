"""Public package reorganization must preserve original tensor/class identity."""
import torch

from adeseg.adapters.high_resolution_skip import MedicalHighResolutionAdapter
from adeseg.adapters.multiscale_channel_attention import (
    MultistageFeatureAggregator,
    MultiscaleChannelValueRefinerV1,
)
from adeseg.backbones.medsam2 import MEDSAM2_LATEST_SHA256
from adeseg.backbones.lean_sam2 import LEAN_AUTHOR_REVISION
from adeseg.losses.teacher_kl import (
    TeacherPreservingKLLoss,
    bernoulli_teacher_kl,
)
from adeseg.losses.segmentation import (
    cmsa_boundary_loss,
    robust_bce_dice_loss,
    supervised_mask_teacher_loss,
)
from adeseg.losses.foreground_preservation import positive_preservation_penalty
from adeseg.inference.medical_decoder_integration import attach_medical_highres_adapter
from adeseg.inference.lean_memory_routing import route_channel_value_refinement
from adeseg.memory.temporal_anchors import multiscale_anchor_score
from adeseg.memory.spatial_position_encoding import indexed_spatial_rope_mps
from adeseg.evaluation.segmentation_metrics import calculate_scores
from modeling.medical_highres.adapter import MedicalHighResolutionAdapter as OriginalMedical
from modeling.medical_highres.teacher_loss import bernoulli_teacher_kl as OriginalKL
from modeling.feature_aggregation import MultistageFeatureAggregator as OriginalLean


def test_new_names_do_not_duplicate_weights_or_models():
    assert MedicalHighResolutionAdapter is OriginalMedical
    assert MultistageFeatureAggregator is OriginalLean
    assert issubclass(MultistageFeatureAggregator, MultiscaleChannelValueRefinerV1)
    assert len(MedicalHighResolutionAdapter().state_dict()) == 10
    assert len(MEDSAM2_LATEST_SHA256) == 64
    assert len(LEAN_AUTHOR_REVISION) == 40


def test_teaching_loss_is_a_training_only_detached_objective():
    native = torch.randn(1, 1, 3, 3, requires_grad=True)
    refined = torch.randn(1, 1, 3, 3, requires_grad=True)
    loss = TeacherPreservingKLLoss()(refined, native)
    assert torch.equal(loss, bernoulli_teacher_kl(refined, native))
    assert bernoulli_teacher_kl is OriginalKL
    loss.backward()
    assert native.grad is None
    assert torch.isfinite(refined.grad).all()


def test_inference_and_loss_exports_exist():
    assert callable(attach_medical_highres_adapter)
    assert callable(route_channel_value_refinement)
    assert callable(multiscale_anchor_score)
    assert callable(indexed_spatial_rope_mps)
    assert callable(calculate_scores)
    assert callable(cmsa_boundary_loss)
    assert callable(robust_bce_dice_loss)
    assert callable(supervised_mask_teacher_loss)
    assert callable(positive_preservation_penalty)
