"""Stage 5: foreground/background feature separation without any GT labels."""
from torch.nn import functional as F
import torch


class ForegroundBackgroundContrastBlock:
    """Derive a scalar spatial contrast from *predicted* soft foreground."""

    @staticmethod
    def forward(context, foreground_probability, *, mode, output_size):
        p = foreground_probability
        fg = (context * p).sum((-2, -1), keepdim=True) / p.sum(
            (-2, -1), keepdim=True).clamp_min(1e-6)
        bg_probability = 1 - p
        bg = (context * bg_probability).sum(
            (-2, -1), keepdim=True) / bg_probability.sum(
                (-2, -1), keepdim=True).clamp_min(1e-6)
        fg_similarity = F.cosine_similarity(context, fg, dim=1, eps=1e-6)
        bg_similarity = F.cosine_similarity(context, bg, dim=1, eps=1e-6)
        if mode in ("mask_only", "diagonal", "global", "foreground_background_no_contrast"):
            contrast = torch.zeros_like(fg_similarity)
        elif mode == "foreground":
            contrast = fg_similarity
        else:
            contrast = fg_similarity - bg_similarity
        return F.interpolate(
            contrast[:, None], size=output_size, mode="bilinear", align_corners=False)
