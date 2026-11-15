"""Read-only projected Q/K spatial-context scores for fixed SAM2 memory tiles.

This is a pooled single-scale SAM2 diagnostic, not CMSA-Net CMA. Scores are
similarities, not learned or measured causal utilities.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr


def _tile_mean(grid):
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] % 4:
        raise ValueError('Expected a square grid divisible into four tiles per side')
    side = grid.shape[0] // 4
    return grid.reshape(4, side, 4, side).mean(dim=(1, 3)).reshape(-1)


def _grid(tokens):
    if tokens.ndim != 2:
        raise ValueError('Expected [spatial tokens, features]')
    side = math.isqrt(tokens.shape[0])
    if side * side != tokens.shape[0] or side % 4:
        raise ValueError('Expected a square spatial grid divisible by four')
    if not torch.isfinite(tokens).all():
        raise ValueError('Nonfinite projected features')
    return tokens.float().T.reshape(1, tokens.shape[1], side, side)


def multiscale_scores(query, key):
    """Return per-tile fine and 2x/4x context scores in native projected space."""
    if query.shape != key.shape:
        raise ValueError('Query and selected memory maps must align in shape')
    q, k = _grid(query), _grid(key)
    scores = {}
    for factor in (1, 2, 4):
        qf = q if factor == 1 else F.avg_pool2d(q, factor, factor)
        kf = k if factor == 1 else F.avg_pool2d(k, factor, factor)
        similarity = F.cosine_similarity(qf, kf, dim=1, eps=1e-8).unsqueeze(1)
        if factor != 1:
            similarity = F.interpolate(similarity, size=q.shape[-2:], mode='nearest')
        scores[f'scale_{factor}'] = _tile_mean(similarity[0, 0]).double().numpy()
    ranked = np.stack([rankdata(scores[f'scale_{factor}'], method='average') for factor in (1, 2, 4)])
    scores['multiscale'] = ranked.mean(axis=0)
    return scores


def attention_tile_mass(q, k, start, stop, *, chunk_size=32):
    """Exact native-softmax tile mass with all spatial and pointer keys in denominator."""
    if q.ndim != 4 or k.ndim != 4 or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError('Expected matching batch/head projected Q/K')
    if q.shape[0] != 1 or not 0 <= start < stop <= k.shape[-2] or stop - start != q.shape[-2]:
        raise ValueError('Expected one square source map aligned to query grid')
    if chunk_size <= 0 or not torch.isfinite(q).all() or not torch.isfinite(k).all():
        raise ValueError('Invalid attention inputs')
    side = math.isqrt(stop - start)
    if side * side != stop - start or side % 4:
        raise ValueError('Invalid source grid')
    mass = torch.zeros(stop - start, dtype=torch.float64, device=q.device)
    for first in range(0, q.shape[-2], chunk_size):
        scores = q[:, :, first:first + chunk_size] @ k.transpose(-1, -2)
        weights = (scores / math.sqrt(q.shape[-1])).softmax(dim=-1)
        mass += weights[..., start:stop].double().sum(dim=(0, 1, 2))
    mass /= q.shape[0] * q.shape[1] * q.shape[-2]
    return (_tile_mean(mass.reshape(side, side)) * (side // 4) ** 2).cpu().numpy()


def evaluate_scores(scores, utility):
    """Within-read descriptive association and fixed lowest-score removal contrast."""
    scores, utility = np.asarray(scores, dtype=float), np.asarray(utility, dtype=float)
    if scores.shape != (16,) or utility.shape != (16,) or not np.isfinite(scores).all() or not np.isfinite(utility).all():
        raise ValueError('Expected 16 finite scores and utilities')
    rho = None
    if np.ptp(scores) > 0 and np.ptp(utility) > 0:
        rho = float(spearmanr(scores, utility).statistic)
    chosen = scores == scores.min()
    return {'spearman_rho': rho, 'selected_tile_ids': np.flatnonzero(chosen).tolist(),
            'selected_removal_utility': float(utility[chosen].mean()),
            'uniform_removal_utility': float(utility.mean()),
            'selected_minus_uniform_dice': float(utility.mean() - utility[chosen].mean())}
