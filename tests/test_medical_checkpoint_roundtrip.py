"""Verify that a training-style adapter checkpoint can be safely loaded."""
import hashlib
from pathlib import Path
import pytest
import torch
from modeling.medical_highres.adapter import MedicalHighResolutionAdapter
from modeling.medical_highres.checkpoints import (
    FORMAT, load_reconstructed_medical_adapter, sha256,
)

def save(tmp_path, *, backbone="a"*64, delta=0.2):
    adapter = MedicalHighResolutionAdapter(maximum_delta=delta)
    with torch.no_grad():
        adapter.output.bias.fill_(0.12)
    record = {
        "format": FORMAT, "reconstruction_not_historical_forward": True,
        "backbone_sha256": backbone, "model_config": "toy-config",
        "source_sha256": sha256(Path(__file__).resolve().parents[1] /
                                "modeling/medical_highres/adapter.py"),
        "training_steps": 3, "maximum_delta": delta,
        "state_dict": adapter.state_dict()
    }
    path = tmp_path / "adapter.pt"
    torch.save(record, path)
    return path, adapter

def test_reconstructed_checkpoint_roundtrip(tmp_path):
    path, source = save(tmp_path)
    model, record = load_reconstructed_medical_adapter(
        path, expected_sha256=sha256(path),
        backbone_sha256="a"*64, model_config="toy-config")
    assert model.checkpoint_sha256 == sha256(path)
    assert not model.training
    early = torch.randn(1,32,16,16)
    deep = torch.randn(1,256,4,4)
    native = torch.randn(1,32,16,16)
    assert torch.equal(source(early,deep,native), model(early,deep,native))

def test_rejects_wrong_backbone_and_source(tmp_path):
    path, _ = save(tmp_path)
    with pytest.raises(ValueError, match="backbone"):
        load_reconstructed_medical_adapter(path, expected_sha256=sha256(path),
                                           backbone_sha256="b"*64, model_config="toy-config")
    with pytest.raises(ValueError, match="SHA256"):
        load_reconstructed_medical_adapter(path, expected_sha256="0"*64,
                                           backbone_sha256="a"*64, model_config="toy-config")
