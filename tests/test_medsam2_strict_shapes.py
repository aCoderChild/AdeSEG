"""Strict checkpoint tensor geometry: fail closed before mutating the model."""
import sys
from pathlib import Path
import torch
from torch import nn
import pytest

# Mirror the supported factory's vendored SAM2 config/import registration.
ROOT = Path(__file__).resolve().parents[1]
SAM2_ROOT = str(ROOT / "external" / "MedSAM2")
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)
from sam2.build_sam import _load_checkpoint


class TinyMedModel(nn.Module):
    def __init__(self, num_maskmem=2):
        super().__init__()
        self.num_maskmem = num_maskmem
        self.linear = nn.Linear(3, 2)
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(num_maskmem))

def save(path, state):
    torch.save({"model": state}, path)
    return path

def test_strict_checkpoint_load_matches_every_tensor_shape(tmp_path):
    source, destination = TinyMedModel(), TinyMedModel()
    path = save(tmp_path / "exact.pt", source.state_dict())
    _load_checkpoint(destination, path)
    for key, tensor in source.state_dict().items():
        assert tensor.shape == destination.state_dict()[key].shape
        assert torch.equal(tensor, destination.state_dict()[key])

def test_unexpected_shape_fails_before_partial_loading(tmp_path):
    model = TinyMedModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    broken = dict(TinyMedModel().state_dict())
    broken["linear.weight"] = torch.randn(3, 3)
    with pytest.raises(RuntimeError, match="incompatible checkpoint tensor shapes"):
        _load_checkpoint(model, save(tmp_path / "bad_shape.pt", broken))
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[key]), f"Partially loaded: {key}"

def test_only_unused_no_memory_position_tensor_may_mismatch(tmp_path):
    model = TinyMedModel(num_maskmem=0)
    source = TinyMedModel(num_maskmem=4)
    path = save(tmp_path / "no_memory.pt", source.state_dict())
    _load_checkpoint(model, path)
    assert model.maskmem_tpos_enc.numel() == 0
    assert tuple(model.linear.weight.shape) == tuple(source.linear.weight.shape)
    assert torch.equal(model.linear.weight, source.linear.weight)
    assert torch.equal(model.linear.bias, source.linear.bias)

def test_no_memory_does_not_waive_other_shape_mismatches(tmp_path):
    model = TinyMedModel(num_maskmem=0)
    broken = dict(TinyMedModel(num_maskmem=4).state_dict())
    broken["linear.bias"] = torch.randn(4)
    with pytest.raises(RuntimeError, match="linear.bias"):
        _load_checkpoint(model, save(tmp_path / "no_memory_bad.pt", broken))

@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_mismatched_checkpoint_key_set_rejected(tmp_path, change):
    model = TinyMedModel()
    values = dict(model.state_dict())
    if change == "missing":
        values.pop("linear.bias")
    else:
        values["extra.weight"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="checkpoint keys differ"):
        _load_checkpoint(model, save(tmp_path / f"{change}.pt", values))

def test_malformed_checkpoint_rejected(tmp_path):
    torch.save({"wrong": {}}, tmp_path / "malformed.pt")
    with pytest.raises(ValueError, match="state dictionary"):
        _load_checkpoint(TinyMedModel(), tmp_path / "malformed.pt")
