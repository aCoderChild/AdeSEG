import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MaskRGBEncoder, MaskRGBEncoderSO, Score


class ReliabilityScoreNet(nn.Module):
    def __init__(self, single_object=True, pretrained=True):
        super().__init__()
        self.single_object = single_object
        self.encoder = (
            MaskRGBEncoderSO(pretrained=pretrained)
            if single_object
            else MaskRGBEncoder(pretrained=pretrained)
        )
        self.score = Score(1024, output_activation="sigmoid")

    def forward(self, image, mask):
        if self.single_object:
            feat = self.encoder(image, mask)
        else:
            other_mask = torch.zeros_like(mask)
            feat = self.encoder(image, mask, other_mask)
        feat = F.interpolate(feat, [24, 24], mode="bilinear", align_corners=False)
        return self.score(feat)
