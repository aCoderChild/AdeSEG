"""CPU test that only the medical adapter receives gradients during training."""
import pytest
import torch
from torch import nn

from modeling.medical_highres import MedicalHighResolutionAdapter
from training.medical_highres_training import (
    medical_training_step, validate_decoder_capture,
)


class ToyNativeDecoder(nn.Module):
    def forward(self, *, image_embeddings, image_pe,
                sparse_prompt_embeddings, dense_prompt_embeddings,
                multimask_output, repeat_image, high_res_features):
        logits = high_res_features[0].mean(dim=1, keepdim=True)
        return logits, None


def capture():
    torch.manual_seed(33)
    return {
        "image_embeddings": torch.randn(1, 256, 4, 4),
        "image_pe": torch.randn(1, 256, 4, 4),
        "sparse_prompt_embeddings": torch.randn(1, 2, 256),
        "dense_prompt_embeddings": torch.randn(1, 256, 4, 4),
        "high_res_0": torch.randn(1, 32, 16, 16),
        "high_res_1": torch.randn(1, 64, 8, 8),
        "target": torch.ones(1, 1, 64, 64),
    }


def test_frozen_decoder_and_trainable_adapter():
    example = capture()
    validate_decoder_capture(example)
    decoder = ToyNativeDecoder().eval()
    adapter = MedicalHighResolutionAdapter(maximum_delta=0.1).train()
    loss, details = medical_training_step(
        decoder, adapter, example, teacher_weight=0.1)
    assert torch.isfinite(loss)
    assert details["teacher_kl"].abs() < 1e-6
    loss.backward()
    assert adapter.output.weight.grad is not None
    assert torch.isfinite(adapter.output.weight.grad).all()
    assert adapter.output.weight.grad.abs().sum().item() > 0


def test_capture_rejects_missing_and_out_of_range_targets():
    example = capture()
    example.pop("target")
    with pytest.raises(ValueError, match="exactly"):
        validate_decoder_capture(example)
    example = capture()
    example["target"].fill_(2.0)
    with pytest.raises(ValueError, match="target"):
        validate_decoder_capture(example)
