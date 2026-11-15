import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from modeling.signed_memory_feature_attribution import SpatialMemoryScalar


ROOT = Path(__file__).resolve().parents[1]
CAPTUM = Path(json.loads((ROOT / "protocols/signed_memory_feature_attribution_design_v1.json")
                        .read_text())["captum_root"])


class Model:
    training = False

    def memory_attention(self, memory, num_obj_ptr_tokens):
        assert num_obj_ptr_tokens == 2
        return memory[:4] + memory[4:].sum(dim=0, keepdim=True)

    def sam_mask_decoder(self, image_embeddings):
        image = image_embeddings
        return (image[:, :1], torch.ones(1, 1, dtype=image.dtype),
                torch.zeros(1, 2, dtype=image.dtype), image.sum().reshape(1, 1))


def fixture():
    memory = torch.arange(-4., 8., dtype=torch.float64).reshape(6, 1, 2)
    model = Model()
    image = model.memory_attention(memory, 2).permute(1, 2, 0).reshape(1, 2, 2, 2)
    capture = SimpleNamespace(memory={"memory": memory, "num_obj_ptr_tokens": 2},
                              decoder={"image_embeddings": image},
                              outputs=model.sam_mask_decoder(image), layers=[0, 1, 2, 3])
    return model, capture


def test_author_input_gradient_signed_exact_and_pointer_excluded():
    model, capture = fixture()
    before = capture.memory["memory"].clone()
    forward = SpatialMemoryScalar(model, capture)
    attribution = forward.attribute(CAPTUM, "object_presence")
    assert attribution.shape == (1, 4, 2)
    assert torch.equal(attribution, forward.inputs)  # Every spatial derivative is one.
    assert attribution.min() < 0
    assert torch.equal(capture.memory["memory"], before)
    assert torch.equal(forward.pointer_tail, before[4:])
    delta = 1e-4
    plus = forward(forward.inputs * (1 + delta), "object_presence")
    minus = forward(forward.inputs * (1 - delta), "object_presence")
    torch.testing.assert_close((plus-minus)/(2*delta), attribution.sum().reshape(1),
                               rtol=1e-9, atol=1e-9)


def test_native_outputs_and_fixed_support_target():
    model, capture = fixture()
    forward = SpatialMemoryScalar(model, capture)
    for expected, actual in zip(capture.outputs, forward.raw_outputs(forward.inputs)):
        assert torch.equal(expected, actual)
    attribution = forward.attribute(CAPTUM, "native_mask_support")
    assert torch.equal(attribution[0, :, 1], torch.zeros(4, dtype=torch.float64))
    assert torch.equal(attribution[0, :, 0], forward.inputs[0, :, 0] / 4)
    with pytest.raises(ValueError, match="dimensions"):
        forward.raw_outputs(forward.inputs[:, :2])


def test_empty_native_support_is_unavailable_not_replaced():
    model, capture = fixture()
    capture.outputs = (-torch.ones_like(capture.outputs[0]), *capture.outputs[1:])
    forward = SpatialMemoryScalar(model, capture)
    with pytest.raises(ValueError, match="unavailable"):
        forward.attribute(CAPTUM, "native_mask_support")


def test_feature_zeroing_is_not_attention_entry_deletion():
    # One-dimensional native attention: zero content retains its normalization
    # entry; deleting that token changes the remaining attention weights.
    query = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    key = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    value = torch.tensor([2., 4.], dtype=torch.float64).reshape(1, 1, 2, 1)
    zeroed = value.clone(); zeroed[..., 0, :] = 0
    sdpa = torch.nn.functional.scaled_dot_product_attention
    assert sdpa(query, key, zeroed).item() == 2
    assert sdpa(query, key[..., 1:, :], value[..., 1:, :]).item() == 4
