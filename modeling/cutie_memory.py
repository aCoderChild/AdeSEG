"""Cutie foreground/background summaries for SAM2 spatial memory experiments."""
import hashlib
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from modeling.cfbi_region_gate import RegionGate

CUTIE_REVISION = 'ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a'
SOURCE_SHA256 = 'bdc14d5bcd39de21ac8105a9e52e0fb7de584144c5ef8628ff669b30df3379bb'


def author_summarizer():
    root = Path(__file__).resolve().parents[1] / 'external' / 'Cutie'
    source = root / 'cutie/model/transformer/object_summarizer.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Cutie summarizer does not match pinned author source')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cutie.model.transformer.object_summarizer import ObjectSummarizer
    return ObjectSummarizer


class CutieMemorySummary(nn.Module):
    """Apply the author module to SAM2 memory [B,1,64,H,W]."""
    def __init__(self):
        super().__init__()
        cfg = OmegaConf.create({'value_dim': 64, 'pixel_pe_scale': 32,
                                'pixel_pe_temperature': 128,
                                'object_summarizer': {'embed_dim': 64, 'num_summaries': 2, 'add_pe': False}})
        self.summarizer = author_summarizer()(cfg)

    def forward(self, mask, memory):
        if memory.ndim != 5 or memory.shape[1:3] != (1, 64) or mask.ndim != 4 or mask.shape[:2] != memory.shape[:2]:
            raise ValueError('Expected mask [B,1,H,W] and memory [B,1,64,H,W]')
        return self.summarizer(mask, memory)[0]

    def three_regions(self, labels, memory):
        """Return Cutie foreground-query summaries for labels 0, 1 and 2."""
        if labels.ndim != 3 or memory.ndim != 5 or labels.shape != (memory.shape[0], *memory.shape[-2:]):
            raise ValueError('Expected labels [B,H,W] aligned with memory')
        if not torch.all((labels == 0) | (labels == 1) | (labels == 2)):
            raise ValueError('Expected labels 0, 1 and 2')
        masks = torch.stack([(labels == i).to(memory.dtype) for i in range(3)], dim=1)
        batch = memory.shape[0]
        values = memory.unsqueeze(1).expand(-1, 3, -1, -1, -1, -1).reshape(batch * 3, *memory.shape[1:])
        summaries = self.forward(masks.reshape(batch * 3, 1, *labels.shape[-2:]), values)
        return summaries[:, 0, 0].reshape(batch, 3, -1)


class CutieRegionGate(RegionGate):
    """Use learned Cutie region summaries with the identity-initialized CFBI gate."""
    def __init__(self):
        super().__init__()
        self.memory_summary = CutieMemorySummary()

    def region_context(self, vectors, labels, mode):
        if vectors.ndim != 2 or vectors.shape[-1] != 64 or labels.shape != vectors.shape[:1]:
            raise ValueError('Expected memory [N,64] and labels [N]')
        if mode not in ('regional', 'generic'):
            raise ValueError('Expected regional or generic context')
        if mode == 'generic':
            labels = torch.zeros_like(labels)
        memory = vectors.T.reshape(1, 1, 64, 1, -1)
        summaries = self.memory_summary.three_regions(labels.reshape(1, 1, -1), memory)
        means = summaries[..., :-1] / summaries[..., -1:].clamp_min(1e-6)
        if mode == 'generic':
            means = means[:, :1].expand(-1, 3, -1)
        return means.reshape(1, 192)
