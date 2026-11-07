import torch
from torch import nn

from modeling.memsam_memory_writer import NativeMemoryWriter, SoftMaskResidualWriter


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


def test_zero_initialized_writer_is_exact_identity_and_keeps_native_mask():
    torch.manual_seed(7)
    model = _Model().requires_grad_(False)
    writer = SoftMaskResidualWriter(channels=8, mid_channels=4)
    features = torch.randn(2, 8, 16, 16)
    masks = torch.rand(2, 1, 64, 64)
    native = model.memory_encoder(features, masks, skip_mask_sigmoid=True)

    with NativeMemoryWriter(model, writer, "regional") as observer:
        adapted = model.memory_encoder(features, masks, skip_mask_sigmoid=True)

    assert torch.equal(adapted["vision_features"], native["vision_features"])
    assert torch.equal(adapted["mask_seen"], native["mask_seen"])
    assert adapted["skip_mask_sigmoid"] is True
    assert observer.records == [{
        "feature_shape": [2, 8, 16, 16],
        "mask_shape": [2, 1, 64, 64],
        "mode": "regional",
    }]
    # Hook teardown is explicit and subsequent calls are native again.
    after = model.memory_encoder(features, masks, skip_mask_sigmoid=True)
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

