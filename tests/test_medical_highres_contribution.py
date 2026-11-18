"""CPU tests of the reconstructed medical adapter and decoder hook."""
from types import SimpleNamespace
import hashlib
import pytest
import torch
from torch import nn

from modeling.medical_highres import (
    MedicalHighResolutionAdapter, attach_medical_highres_adapter,
    bernoulli_teacher_kl, supervised_mask_teacher_loss,
)


def features():
    return (torch.randn(2, 32, 16, 16),
            torch.randn(2, 256, 4, 4),
            torch.randn(2, 32, 16, 16))


def test_saved_five_layer_schema_and_exact_zero_identity():
    torch.manual_seed(42)
    model = MedicalHighResolutionAdapter()
    assert len(model.state_dict()) == 10
    assert list(model.state_dict()) == list(model.KEY_NAMES)
    e, d, skip = features()
    assert torch.equal(model(e, d, skip), skip)
    assert model.output.weight.shape == (32, 32, 1, 1)
    assert model.spatial.weight.shape == (32, 1, 3, 3)


def test_residual_bounded_and_gradients():
    model = MedicalHighResolutionAdapter(maximum_delta=0.2)
    with torch.no_grad():
        model.output.weight.normal_(mean=0, std=0.3)
    early, deep, skip = features()
    output = model(early, deep, skip)
    assert (output - skip).abs().max().item() <= 0.200001
    output.square().mean().backward()
    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all()


def test_native_teacher_kl_and_supervised_loss():
    teacher = torch.tensor([[[[3., -5.], [1., -1024.]]]])
    student = teacher.clone().requires_grad_(True)
    assert abs(bernoulli_teacher_kl(student, teacher).item()) < 5e-6
    target = torch.tensor([[[[1., 0.], [1., 0.]]]])
    loss, details = supervised_mask_teacher_loss(student, teacher, target)
    assert loss.isfinite()
    assert set(details) == {
        "bce", "dice", "teacher_kl", "positive_preservation"}
    loss.backward()
    assert student.grad.isfinite().all()
    assert teacher.grad is None


class MinimalDecoder(nn.Module):
    use_high_res_features = True

    def forward(self, image_embeddings, image_pe=None,
                sparse_prompt_embeddings=None, dense_prompt_embeddings=None,
                multimask_output=False, repeat_image=False,
                high_res_features=None):
        return high_res_features[0], high_res_features[1]


def test_decoder_hook_scoped_and_second_skip_preserved():
    predictor = SimpleNamespace(sam_mask_decoder=MinimalDecoder())
    adapter = MedicalHighResolutionAdapter(maximum_delta=0.2)
    early, deep, skip = features()
    kwargs = {"image_embeddings": deep, "high_res_features": [skip, early]}
    original = predictor.sam_mask_decoder(**kwargs)
    with attach_medical_highres_adapter(predictor, adapter, allow_untrained=True):
        assert torch.equal(predictor.sam_mask_decoder(**kwargs)[0], original[0])
        with pytest.raises(RuntimeError, match="already installed"):
            with attach_medical_highres_adapter(
                    predictor, adapter, allow_untrained=True):
                pass
        with torch.no_grad():
            adapter.output.bias.fill_(0.5)
        changed = predictor.sam_mask_decoder(**kwargs)
        assert not torch.equal(changed[0], original[0])
        assert torch.equal(changed[1], original[1])
    assert torch.equal(predictor.sam_mask_decoder(**kwargs)[0], original[0])
    assert not hasattr(
        predictor.sam_mask_decoder, "_medical_highres_adapter_active")


def test_strict_checkpoint_bytes_and_state_names(tmp_path):
    adapter = MedicalHighResolutionAdapter()
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "historical", "state_dict": adapter.state_dict()}, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    fresh = MedicalHighResolutionAdapter()
    record = fresh.load_historical_state(
        path, expected_sha256=sha, allowed_formats={"historical"})
    assert record["format"] == "historical"
    assert fresh.checkpoint_sha256 == sha
    with pytest.raises(ValueError, match="mismatch"):
        fresh.load_historical_state(path, expected_sha256="0" * 64)
