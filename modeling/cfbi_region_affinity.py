"""CFBI nearest-neighbour regional evidence adapted to SAM2 memory reads."""

import ast
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from modeling.region_tokens import _verified_source

CFBI_REVISION = "91fc6a31b1752a8a9110b3ed480e05656f756691"
MATCHING_SOURCE = "networks/layers/matching.py"
GRID = 64
MATCH_GRID = 16
POOL = GRID // MATCH_GRID


@lru_cache(maxsize=1)
def author_global_matching():
    path, source = _verified_source("CFBI", CFBI_REVISION, MATCHING_SOURCE)
    names = {
        "_pairwise_distances",
        "_flattened_pairwise_distances",
        "_nn_features_per_object_for_chunk",
        "_nearest_neighbor_features_per_object_in_chunks",
        "global_matching",
    }
    tree = ast.parse(source)
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("Pinned CFBI matching functions are incomplete")
    constant = next(node for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name)
                            and target.id == "WRONG_LABEL_PADDING_DISTANCE"
                            for target in node.targets))
    namespace = {"np": np, "torch": torch, "F": F,
                 "WRONG_LABEL_PADDING_DISTANCE": ast.literal_eval(constant.value)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["global_matching"]


def regional_matching_evidence(query, memory, labels, *, chunks=32):
    """Return three author CFBI matching maps from every spatial memory frame."""
    if query.shape != (1, GRID * GRID, 256):
        raise ValueError("Expected one 64x64 SAM2 query grid with 256 channels")
    if memory.ndim != 3 or memory.shape[0] == 0 or memory.shape[0] % (GRID * GRID):
        raise ValueError("Memory must contain complete 64x64 spatial frames")
    if memory.shape[1:] != (1, 256):
        raise ValueError("Expected SAM2 memory [tokens, 1, 256]")
    frame_count = memory.shape[0] // (GRID * GRID)
    if labels.shape != (1, memory.shape[0]) or not torch.all((labels >= 0) & (labels <= 2)):
        raise ValueError("Stored-mask region labels must align with spatial memory")

    query_map = query[0].reshape(GRID, GRID, 256).permute(2, 0, 1)[None]
    query_small = F.avg_pool2d(query_map.float(), POOL, POOL)[0].permute(1, 2, 0).contiguous()
    memory_map = memory[:, 0].reshape(frame_count, GRID, GRID, 256).permute(0, 3, 1, 2).float()
    label_map = labels[0].reshape(frame_count, GRID, GRID)

    pooled_features = []
    pooled_labels = []
    for region in range(3):
        mask = label_map.eq(region).float().unsqueeze(1)
        denominator = F.avg_pool2d(mask, POOL, POOL)
        numerator = F.avg_pool2d(memory_map * mask, POOL, POOL)
        feature = numerator / denominator.clamp_min(1e-6)
        coverage = denominator[:, 0] > 0
        pooled_features.append(feature.permute(0, 2, 3, 1))
        region_labels = torch.zeros((*coverage.shape, 3), device=labels.device, dtype=torch.float32)
        region_labels[..., region] = coverage.float()
        pooled_labels.append(region_labels)

    # CFBI accepts one reference feature grid and one-hot labels. Stack each
    # stored frame and role as rows; no memory frame or original token is dropped.
    reference = torch.stack(pooled_features, dim=1).reshape(-1, MATCH_GRID, 256)
    reference_labels = torch.stack(pooled_labels, dim=1).reshape(-1, MATCH_GRID, 3)
    present = reference_labels.sum(dim=(0, 1)) > 0
    # CFBI's matcher expects compact semantic embeddings. Raw SAM2 projections
    # have much larger norms, making its sigmoid(distance) saturate to one.
    query_small = F.normalize(query_small, dim=-1, eps=1e-6)
    reference = F.normalize(reference, dim=-1, eps=1e-6)
    evidence = author_global_matching()(
        reference.contiguous(), query_small.contiguous(), reference_labels.contiguous(),
        n_chunks=chunks, dis_bias=torch.zeros(3, device=query.device),
        atrous_rate=1, use_float16=False,
    )
    evidence = evidence[0, ..., 0].permute(2, 0, 1).unsqueeze(0)
    evidence = torch.where(present[None, :, None, None], evidence, torch.zeros_like(evidence))
    return evidence, present


class RegionalAffinityHead(nn.Module):
    """Small bounded residual driven by CFBI's role-specific match evidence."""

    def __init__(self, channels=256, max_residual_rms=0.1):
        super().__init__()
        self.projection = nn.Conv2d(3, channels, 1, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.max_residual_rms = float(max_residual_rms)

    def forward(self, native, evidence):
        if native.ndim != 4 or native.shape[1:] != (256, GRID, GRID):
            raise ValueError("Expected native attention map [1, 256, 64, 64]")
        if evidence.shape != (1, 3, MATCH_GRID, MATCH_GRID):
            raise ValueError("Expected three 16x16 regional evidence maps")
        evidence = F.interpolate(evidence, size=(GRID, GRID), mode="bilinear", align_corners=False)
        native32, evidence32 = native.float(), evidence.float()
        scale = native32.square().mean(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        delta = self.projection(evidence32)
        output = native32 + torch.tanh(delta / scale) * (self.max_residual_rms * scale)
        if torch.count_nonzero(self.projection.weight.detach()) == 0:
            output = output - output.detach() + native32.detach()
        return output.to(dtype=native.dtype)


class RegionalAffinityHeads(nn.Module):
    def __init__(self, layers=4, channels=256):
        super().__init__()
        self.layers = nn.ModuleList(RegionalAffinityHead(channels) for _ in range(layers))
