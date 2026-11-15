"""Soft query-to-memory region affinity for dense SAM2 memory attention."""

import torch
import torch.nn.functional as F
from torch import nn

from modeling.region_mass_attention import native_identity


def soft_mixed_cell_roles(probability_256):
    """Return background, foreground and within-cell mixedness on the 64x64 grid."""
    probability = torch.as_tensor(probability_256).detach().float()
    while probability.ndim > 2 and probability.shape[0] == 1:
        probability = probability.squeeze(0)
    if probability.shape != (256, 256) or not torch.isfinite(probability).all():
        raise ValueError("Expected a finite 256x256 probability map")
    if (probability < 0).any() or (probability > 1).any():
        raise ValueError("Mask probabilities must be in [0, 1]")

    cells = probability.reshape(64, 4, 64, 4).permute(0, 2, 1, 3).reshape(64, 64, 16)
    mean = cells.mean(dim=-1)
    variance = cells.var(dim=-1, unbiased=False)
    boundary = (4 * variance).clamp(0, 1)
    foreground = mean * (1 - boundary)
    background = (1 - mean) * (1 - boundary)
    roles = torch.stack((background, foreground, boundary), dim=-1)
    if not torch.allclose(roles.sum(-1), torch.ones_like(mean), atol=1e-6):
        raise RuntimeError("Soft roles must sum to one per memory token")
    return roles


def mask_logits_to_soft_roles(mask_logits):
    logits = torch.as_tensor(mask_logits).detach().float()
    while logits.ndim > 2 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.shape != (256, 256) or not torch.isfinite(logits).all():
        raise ValueError("Expected finite native 256x256 stored-mask logits")
    return soft_mixed_cell_roles(logits.sigmoid())


def target_to_soft_roles(target):
    """Resize a binary training target to SAM2's native mask grid, then label cells."""
    target = torch.as_tensor(target).detach().float()
    while target.ndim > 2 and target.shape[0] == 1:
        target = target.squeeze(0)
    if target.ndim != 2 or not torch.isfinite(target).all():
        raise ValueError("Expected a finite 2-D training mask")
    if (target < 0).any() or (target > 1).any():
        raise ValueError("Training mask must be in [0, 1]")
    probability_256 = F.interpolate(
        target[None, None], size=(256, 256), mode="area"
    )[0, 0]
    return soft_mixed_cell_roles(probability_256)


class SoftRoleAffinityLayer(nn.Module):
    """Apply HODOR-style finite soft mask biases, mixed by predicted query roles."""

    def __init__(self, heads=1, head_dim=256, max_bias=8.0):
        super().__init__()
        self.query_role = nn.Linear(head_dim, 3)
        nn.init.zeros_(self.query_role.weight)
        nn.init.zeros_(self.query_role.bias)
        self.role_bias = nn.Parameter(torch.zeros(heads, 3))
        self.max_bias = float(max_bias)
        self.last_query_logits = None
        self.last_query_roles = None

    def region_context(self, vectors, memberships, mode):
        if mode not in ("regional", "global"):
            raise ValueError("Unknown region-affinity mode")
        self._memory_roles = memberships
        return vectors.new_zeros((vectors.shape[0], 1))

    def forward(self, native, query, key, value, memberships, pointer_tokens,
                mode="regional", *, query_content=None, sdpa=None):
        if mode not in ("regional", "global"):
            raise ValueError("Expected regional or global mode")
        if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
            raise ValueError("Expected [batch, heads, tokens, channels] attention tensors")
        batch, heads, query_count, head_dim = query.shape
        spatial = key.shape[-2] - pointer_tokens
        if not 0 <= pointer_tokens < key.shape[-2] or spatial <= 0:
            raise ValueError("Invalid spatial or pointer key count")
        if memberships.shape != (batch, spatial, 3):
            raise ValueError("Memory memberships must align with spatial keys")
        if not torch.isfinite(memberships).all() or (memberships < 0).any():
            raise ValueError("Memory memberships must be finite and nonnegative")
        if not torch.allclose(
            memberships.sum(-1), torch.ones_like(memberships[..., 0]), atol=1e-5
        ):
            raise ValueError("Memory memberships must sum to one")
        if self.role_bias.shape[0] != heads or self.query_role.in_features != head_dim:
            raise ValueError("Attention head dimensions differ from the registered head")
        if query_content is None or query_content.shape != query.shape:
            raise ValueError(
                "Expected per-head query features for role prediction: "
                f"query={tuple(query.shape)}, content="
                f"{None if query_content is None else tuple(query_content.shape)}"
            )
        if sdpa is None:
            raise ValueError("A scaled dot-product attention function is required")

        query_logits = self.query_role(query_content)
        query_roles = query_logits.softmax(dim=-1)
        self.last_query_logits = query_logits
        self.last_query_roles = query_roles

        if mode == "global":
            memory_roles = memberships.mean(dim=1, keepdim=True).expand(-1, spatial, -1)
        else:
            memory_roles = memberships

        bias_scales = self.max_bias * self.role_bias.tanh()
        role_outputs = []
        for role in range(3):
            # HODOR's soft-mask rule is a finite penalty for missing membership.
            spatial_bias = -bias_scales[:, role][None, :, None, None] * (
                1 - memory_roles[:, None, None, :, role]
            )
            if pointer_tokens:
                pointer_bias = torch.zeros(
                    batch, heads, 1, pointer_tokens,
                    device=spatial_bias.device, dtype=spatial_bias.dtype,
                )
                attention_bias = torch.cat((spatial_bias, pointer_bias), dim=-1)
            else:
                attention_bias = spatial_bias
            role_outputs.append(sdpa(
                query, key, value, attn_mask=attention_bias, dropout_p=0.0
            ))

        mixed = sum(
            output * query_roles[..., role].unsqueeze(-1)
            for role, output in enumerate(role_outputs)
        )
        # Preserve native output bit-for-bit at zero bias while keeping gradients
        # through the soft role branches and bias parameters.
        if torch.count_nonzero(bias_scales.detach()).item() == 0:
            mixed = native_identity(query, key, value, sdpa, mixed)
        valid = (memberships.sum(dim=1) > 0)
        return mixed, valid


class SoftRoleAffinityHeads(nn.Module):
    requires_query_content = True

    def __init__(self, layers=4, heads=1, head_dim=256, max_bias=8.0):
        super().__init__()
        self.layers = nn.ModuleList(
            SoftRoleAffinityLayer(heads, head_dim, max_bias) for _ in range(layers)
        )

