"""Cutie-author regional summaries as an auxiliary SAM2 memory read."""
import hashlib
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from modeling.cutie_memory import CUTIE_REVISION, SOURCE_SHA256, author_summarizer
from modeling.cutie_region_queries import CUTIE_WEIGHT_MD5, author_modules


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CHECKPOINT = ROOT / 'external/Cutie/weights/cutie-base-mega.pth'


def _author_cross_attention():
    author_modules()
    from cutie.model.transformer.transformer_layers import CrossAttention
    return CrossAttention


class CutieRegionPrototypes(nn.Module):
    def __init__(self, checkpoint=SUMMARY_CHECKPOINT):
        super().__init__()
        source = ROOT / 'external/Cutie/cutie/model/transformer/object_summarizer.py'
        revision = subprocess.check_output(
            ['git', '-C', str(ROOT / 'external/Cutie'), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != CUTIE_REVISION:
            raise RuntimeError('Cutie repository revision differs from the pinned source')
        if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise RuntimeError('Cutie object summarizer differs from the pinned author source')
        if hashlib.md5(Path(checkpoint).read_bytes()).hexdigest() != CUTIE_WEIGHT_MD5:
            raise RuntimeError('Cutie checkpoint differs from the verified author weights')

        config = OmegaConf.create({
            'value_dim': 256,
            'pixel_pe_scale': 32,
            'pixel_pe_temperature': 128,
            'object_summarizer': {'embed_dim': 256, 'num_summaries': 16, 'add_pe': True},
        })
        self.summarizer = author_summarizer()(config)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        prefix = 'object_summarizer.'
        weights = {key[len(prefix):]: value for key, value in state.items()
                   if key.startswith(prefix)}
        self.summarizer.load_state_dict(weights, strict=True)
        self.summarizer.requires_grad_(False)

    def forward(self, values, labels):
        """Return Cutie foreground summaries [frames, regions, 8, 256] and areas."""
        if values.ndim != 4 or values.shape[1:] != (256, 64, 64):
            raise ValueError('Expected per-frame projected memory [frames,256,64,64]')
        if labels.shape != (values.shape[0], 64, 64):
            raise ValueError('Stored-mask region labels must align with each memory frame')
        if not torch.isin(labels, torch.tensor([0, 1, 2], device=labels.device)).all():
            raise ValueError('Expected background, foreground and boundary labels')

        frames = values.shape[0]
        masks = torch.stack([(labels == region).to(values.dtype) for region in range(3)], dim=1)
        masks = masks.reshape(frames * 3, 1, 64, 64)
        object_values = values[:, None].expand(-1, 3, -1, -1, -1).reshape(
            frames * 3, 1, 256, 64, 64)
        summaries, _ = self.summarizer(masks, object_values)
        summaries = summaries[:, 0]
        features = summaries[:, :8, :256].reshape(frames, 3, 8, 256)
        areas = summaries[:, :8, 256].reshape(frames, 3, 8)
        features = features / areas.clamp_min(1e-6)[..., None]
        return features, areas


class CutieRegionMessageHeads(nn.Module):
    def __init__(self, checkpoint=SUMMARY_CHECKPOINT, layer_index=0):
        super().__init__()
        CrossAttention = _author_cross_attention()
        self.prototypes = CutieRegionPrototypes(checkpoint)
        self.reader = CrossAttention(
            256, 8, add_pe_to_qkv=[True, True, False], residual=False, norm=True)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        prefix = 'object_transformer.blocks.0.read_from_pixel.'
        reader_weights = {
            'cross_attn.' + key.removeprefix(prefix + 'cross_attn.'): value
            for key, value in state.items() if key.startswith(prefix + 'cross_attn.')
        }
        reader_weights.update({
            key.removeprefix(prefix): value
            for key, value in state.items() if key in (prefix + 'norm.weight', prefix + 'norm.bias')
        })
        self.reader.load_state_dict(reader_weights, strict=True)
        self.layer_index = int(layer_index)
        self.region_position = nn.Parameter(torch.zeros(3, 256))
        self.scale_logit = nn.Parameter(torch.zeros(()))

    def make_messages(self, values, labels, output_dtype):
        if values.ndim != 4 or values.shape[1:] != (256, 64, 64):
            raise ValueError('Expected projected memory values [frames,256,64,64]')
        if labels.shape != (values.shape[0], 64, 64):
            raise ValueError('Predicted-region labels do not align with memory values')
        summaries, areas = self.prototypes(values, labels)
        valid = areas > 0
        messages = summaries[valid].unsqueeze(0)
        region_ids = torch.arange(3, device=labels.device)[None, :, None].expand(
            values.shape[0], -1, 8)[valid]
        positions = self.region_position[region_ids].unsqueeze(0)
        if messages.shape[1] == 0:
            raise RuntimeError('No nonempty regional memory prototypes')
        # The published Cutie attention module is FP32; keep the auxiliary
        # branch numerically stable even when SAM2 runs reduced precision.
        messages = messages.float()
        positions = positions.float()
        diagnostics = {
            'frames': int(values.shape[0]),
            'spatial_keys': int(values.shape[0] * 4096),
            'pointer_keys': 0,
            'region_counts': [(labels == region).sum(dim=(1, 2)).tolist() for region in range(3)],
            'prototype_tokens': int(valid.sum()),
        }
        return messages, positions, diagnostics

    def read(self, query, messages, positions, native_output):
        query = query.to(dtype=self.region_position.dtype)
        query_position = torch.zeros_like(query)
        branch, _ = self.reader(query, messages, query_position, positions)
        scale = self.scale_logit.tanh().to(native_output.dtype)
        return native_output + scale * branch.to(native_output.dtype)

    def forward(self, native, query, key, value, labels, pointer_tokens,
                mode='regional', *, sdpa=None):
        if mode != 'regional':
            raise ValueError('This experiment only supports predicted-region utilization')
        if native.ndim != 4 or query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
            raise ValueError('Expected SAM2 attention tensors [B,heads,tokens,head_dim]')
        batch, heads, query_count, head_dim = query.shape
        spatial = key.shape[-2] - pointer_tokens
        channels = heads * head_dim
        if batch != 1 or channels != 256 or labels.shape != (batch, spatial):
            raise ValueError('Region labels do not align with 256-channel memory keys')
        if spatial <= 0 or spatial % 4096:
            raise ValueError('Expected complete 64x64 memory frames')
        frames = spatial // 4096
        values = value[..., :spatial, :].permute(0, 2, 1, 3).reshape(
            frames, 4096, channels).float()
        values = values.reshape(frames, 64, 64, channels).permute(0, 3, 1, 2).contiguous()
        labels_per_frame = labels.reshape(frames, 64, 64)
        messages, positions, diagnostics = self.make_messages(values, labels_per_frame, value.dtype)
        query_flat = query.permute(0, 2, 1, 3).reshape(batch, query_count, channels)
        native_flat = native.permute(0, 2, 1, 3).reshape(batch, query_count, channels)
        updated = self.read(query_flat, messages, positions, native_flat)
        result = updated.reshape(batch, query_count, heads, head_dim).permute(0, 2, 1, 3)
        valid = torch.stack([labels.eq(region).any(dim=1) for region in range(3)], dim=1)
        return result.to(native.dtype), valid


class CutieRegionSummaryHeads(nn.Module):
    """Read Cutie summaries as an auxiliary branch beside native SAM2 attention."""

    def __init__(self, layers=4, checkpoint=SUMMARY_CHECKPOINT):
        super().__init__()
        self.layers = nn.ModuleList(CutieRegionMessageHeads(checkpoint, layer_index=index)
                                    for index in range(layers))

    # NativeRegionalReadout dispatches directly to one of the four layer heads.
