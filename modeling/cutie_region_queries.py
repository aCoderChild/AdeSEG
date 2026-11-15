"""Cutie query-transformer adapters for SAM2 memory attention."""
import hashlib
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

CUTIE_REVISION = 'ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a'
CUTIE_WEIGHT_MD5 = 'a6071de6136982e396851903ab4c083a'
SOURCE_HASHES = {
    'object_transformer.py': '9013aa47cdc24c1422e7c941e0e64431b55e20961247738fed4831c2fcc80982',
    'transformer_layers.py': 'baff1ff73ca2c96ac8c0b23524b3a8ae380d5c70153f16c78360b4bc76dbc8ef',
    'positional_encoding.py': '691c2b330a93b66ca0534b1e9709cc45c5ce919e6d6d72a84bc7a04ecc80cff2',
}


def author_modules():
    root = Path(__file__).resolve().parents[1] / 'external' / 'Cutie'
    revision = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != CUTIE_REVISION or dirty:
        raise RuntimeError('Cutie repository is not the pinned clean checkout')
    for name, expected in SOURCE_HASHES.items():
        path = root / 'cutie/model/transformer' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Cutie source mismatch: {name}')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cutie.model.transformer.object_transformer import QueryTransformerBlock
    from cutie.model.transformer.positional_encoding import PositionalEncoding
    return QueryTransformerBlock, PositionalEncoding


class CutieRegionQueryCore(nn.Module):
    """One shared Cutie block and feature bridge for SAM2's four layers."""

    def __init__(self, channels=256, heads=8, feedforward=2048, pretrained_checkpoint=None):
        super().__init__()
        block_class, positional_encoding = author_modules()
        cfg = OmegaConf.create({
            'object_transformer': {
                'embed_dim': channels,
                'num_heads': heads,
                'num_queries': 3,
                'ff_dim': feedforward,
                'read_from_pixel': {'add_pe_to_qkv': [True, True, False]},
                'query_self_attention': {'add_pe_to_qkv': [True, True, False]},
                'read_from_query': {'add_pe_to_qkv': [True, True, False], 'output_norm': False},
            },
        })
        self.block = block_class(cfg)
        if pretrained_checkpoint is not None:
            self.load_author_block(pretrained_checkpoint)
        self.memory_projection = nn.Linear(channels, channels, bias=False)
        self.read_projection = nn.Linear(channels, channels, bias=False)
        self.query_projection = nn.Linear(channels, channels, bias=False)
        self.output_projection = nn.Linear(channels, channels, bias=False)
        for projection in (self.memory_projection, self.read_projection,
                           self.query_projection, self.output_projection):
            nn.init.eye_(projection.weight)
        self.pixel_position = positional_encoding(
            channels, scale=32, temperature=128, channel_last=False, transpose_output=True)

    def load_author_block(self, path):
        path = Path(path)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != CUTIE_WEIGHT_MD5:
            raise RuntimeError('Cutie checkpoint does not match the official base model')
        weights = torch.load(path, map_location='cpu', weights_only=True)
        prefix = 'object_transformer.blocks.0.'
        block_weights = {name[len(prefix):]: value for name, value in weights.items()
                         if name.startswith(prefix)}
        self.block.load_state_dict(block_weights, strict=True)
        self.block.requires_grad_(False)


class CutieRegionQueryResidual(nn.Module):
    """Refine the native read with Cutie's pretrained region-query block."""

    def __init__(self, core):
        super().__init__()
        object.__setattr__(self, '_core', core)
        channels = core.memory_projection.in_features
        self.region_queries = nn.Parameter(torch.zeros(1, 3, channels))
        self.region_positions = nn.Parameter(torch.zeros(1, 3, channels))
        self.scale = nn.Parameter(torch.zeros(()))
        self.max_scale = 0.25

    def forward(self, native, query, key, value, labels, pointer_tokens,
                mode='regional', *, sdpa=None):
        core = self._core
        if native.ndim != 4 or value.ndim != 4 or value.shape != key.shape:
            raise ValueError('Expected SAM2 attention tensors [B,heads,tokens,channels]')
        batch, heads, query_count, head_dim = native.shape
        spatial = key.shape[-2] - pointer_tokens
        channels = heads * head_dim
        if channels != core.memory_projection.in_features or labels.shape != (batch, spatial):
            raise ValueError('SAM2 feature width or region map does not match Cutie block')
        if mode not in ('regional', 'generic'):
            raise ValueError('Expected regional or generic mode')
        side = int(query_count ** 0.5)
        if side * side != query_count:
            raise ValueError('Expected a square SAM2 query grid')

        values = value[..., :spatial, :].permute(0, 2, 1, 3).reshape(batch, spatial, channels).float()
        values = core.memory_projection(values)
        if mode == 'generic':
            pooled = values.mean(dim=1, keepdim=True).expand(-1, 3, -1)
        else:
            pooled = torch.stack([
                (values * labels.eq(region)[..., None]).sum(dim=1)
                / labels.eq(region).sum(dim=1, keepdim=True).clamp_min(1)
                for region in range(3)
            ], dim=1)

        native_map = native.permute(0, 2, 1, 3).reshape(batch, query_count, channels)
        query_map = query.permute(0, 2, 1, 3).reshape(batch, query_count, channels)
        pixel = core.read_projection(native_map) + core.query_projection(query_map)
        pixel = pixel.transpose(1, 2).reshape(batch, 1, channels, side, side).float()
        query_tokens = pooled + self.region_queries
        query_positions = self.region_positions.expand(batch, -1, -1)
        pixel_positions = core.pixel_position(pixel.squeeze(1)).flatten(1, 2)
        _, enriched, _, _ = core.block(
            query_tokens, pixel, query_positions, pixel_positions, attn_mask=None)
        delta = (enriched - pixel).flatten(3).squeeze(1).transpose(1, 2)
        delta = core.output_projection(delta)
        output = native_map + self.max_scale * self.scale.tanh() * delta
        output = output.to(native.dtype).reshape(batch, query_count, heads, head_dim).transpose(1, 2)
        valid = torch.stack([labels.eq(region).any(dim=1) for region in range(3)], dim=1)
        return output, valid


class CutieRegionQueryHeads(nn.Module):
    def __init__(self, layers=4, pretrained_checkpoint=None):
        super().__init__()
        self.core = CutieRegionQueryCore(pretrained_checkpoint=pretrained_checkpoint)
        self.layers = nn.ModuleList(CutieRegionQueryResidual(self.core) for _ in range(layers))
