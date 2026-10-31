"""Current-frame ablation of the pinned UACANet-derived source32 head."""
import copy
import torch
from torch import nn
import torch.nn.functional as F

from modeling.regional_refinement import RegionalRefinement


class CurrentFrameRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct the original first: shared blocks have exactly its seeded init.
        original = RegionalRefinement()
        self.project = original.project
        self.fusion = copy.deepcopy(original.head.conv_out2)
        # Remove the 32 context input channels, rather than train dead parameters.
        old = self.fusion.conv
        self.fusion.conv = nn.Conv2d(64, old.out_channels, old.kernel_size,
                                     old.stride, old.padding, bias=False)
        with torch.no_grad():
            self.fusion.conv.weight.copy_(old.weight[:, :64])
        self.refine = original.head.conv_out3
        self.output = original.head.conv_out4

    def forward(self, query, contexts, valid, native_logits, object_score):
        # contexts/valid are intentionally unused, for source32 batch compatibility.
        x = self.refine(self.fusion(self.project(query)))
        correction = F.interpolate(self.output(x), size=native_logits.shape[-2:],
                                   mode='bilinear', align_corners=False)
        return native_logits + correction * (object_score.reshape(-1,1,1,1) > 0)
