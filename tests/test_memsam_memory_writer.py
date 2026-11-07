import torch
from torch import nn

from modeling.memsam_memory_writer import (
    NativeMemoryWriter,
    SoftMaskResidualWriter,
    _tensor_sha256,
)


class _MemoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.25))

    def forward(self, pix_feat, masks, skip_mask_sigmoid=False):
        return {
            "vision_features": pix_feat * self.scale,
            "vision_pos_enc": [torch.zeros_like(pix_feat)],
            "mask_seen": masks,
            "skip_mask_sigmoid": skip_mask_sigmoid,
        }


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_encoder = _MemoryEncoder()
        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0

    def track_step(self, frame_idx, pix_feat, masks, skip_mask_sigmoid=False):
        return self.memory_encoder(pix_feat, masks, skip_mask_sigmoid)

    def _run_memory_encoder(self, inference_state, frame_idx, pix_feat, masks,
                            skip_mask_sigmoid=False):
        # Mirrors SAM2's preflight call shape: the prompt-frame memory write
        # reaches the encoder directly with inference state and frame index.
        return self.memory_encoder(pix_feat, masks, skip_mask_sigmoid)


def test_zero_initialized_writer_is_exact_identity_and_keeps_native_mask():
    torch.manual_seed(7)
    model = _Model().requires_grad_(False)
    writer = SoftMaskResidualWriter(channels=8, mid_channels=4)
    features = torch.randn(2, 8, 16, 16)
    probabilities = torch.rand(2, 1, 64, 64)
    memory_mask = probabilities * model.sigmoid_scale_for_mem_enc + model.sigmoid_bias_for_mem_enc
    native = model.memory_encoder(features, memory_mask, skip_mask_sigmoid=True)
    recovered = []
    writer.register_forward_pre_hook(
        lambda _module, args: recovered.append(args[1].detach().clone())
    )

    with NativeMemoryWriter(model, writer, "regional") as observer:
        adapted = model.track_step(17, features, memory_mask, skip_mask_sigmoid=True)
        preflight = model._run_memory_encoder(
            {"prompt": True}, 3, features, memory_mask, skip_mask_sigmoid=True
        )

    assert torch.equal(adapted["vision_features"], native["vision_features"])
    assert torch.equal(preflight["vision_features"], native["vision_features"])
    assert torch.equal(adapted["mask_seen"], memory_mask)
    assert torch.equal(preflight["mask_seen"], memory_mask)
    assert torch.equal(adapted["mask_seen"], native["mask_seen"])
    assert adapted["skip_mask_sigmoid"] is True
    assert preflight["skip_mask_sigmoid"] is True
    assert len(recovered) == 2
    assert all(torch.allclose(item, probabilities, atol=2e-7, rtol=2e-7) for item in recovered)
    assert [record["frame"] for record in observer.records] == [17, 3]
    record = observer.records[0]
    assert record["frame"] == 17
    assert record["feature_shape"] == [2, 8, 16, 16]
    assert record["mask_shape"] == [2, 1, 64, 64]
    assert record["mode"] == "regional"
    assert record["memory_encoder_mask_sha256"] == _tensor_sha256(memory_mask)
    assert record["recovered_probability_sha256"] == _tensor_sha256(recovered[0])
    assert record["memory_mask_scale"] == 20.0
    assert record["memory_mask_bias"] == -10.0
    assert all(len(record[key]) == 64 for key in (
        "input_feature_sha256", "memory_encoder_mask_sha256",
        "recovered_probability_sha256", "adapted_feature_sha256"
    ))
    # Hook teardown is explicit and subsequent calls are native again.
    after = model.memory_encoder(features, memory_mask, skip_mask_sigmoid=True)
    assert torch.equal(after["vision_features"], native["vision_features"])


def test_writer_learns_from_soft_mask_but_generic_control_is_mask_neutral():
    torch.manual_seed(11)
    regional = SoftMaskResidualWriter(channels=8, mid_channels=4)
    generic = SoftMaskResidualWriter(channels=8, mid_channels=4)
    generic.load_state_dict(regional.state_dict())
    with torch.no_grad():
        for writer in (regional, generic):
            writer.author_path.conv_1x1.weight.fill_(0.01)
            writer.author_path.conv_1x1.bias.fill_(0.02)

    features = torch.randn(1, 8, 12, 12)
    mask = torch.zeros(1, 1, 48, 48)
    mask[:, :, 12:36, 12:36] = 0.9
    regional_out = regional(features, mask, "regional")
    generic_out = generic(features, mask, "generic")
    assert regional_out.shape == features.shape
    assert generic_out.shape == features.shape
    assert not torch.equal(regional_out, generic_out)

    loss = regional_out.square().mean()
    loss.backward()
    assert regional.author_path.conv_1x1.weight.grad is not None
    assert torch.isfinite(regional.author_path.conv_1x1.weight.grad).all()
    assert regional.author_path.conv_1x1.weight.grad.abs().sum() > 0

