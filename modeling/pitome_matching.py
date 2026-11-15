"""Thin adapter for the pinned PiToMe author matcher; engineering use only."""
from contextlib import contextmanager
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import torch

PITOME_REVISION = '550b5deed94aadfeac28bfbe381d87e672044a40'
PITOME_SOURCE = 'algo/pitome/merge.py'


@lru_cache(maxsize=1)
def author_pitome(repository):
    repository = Path(repository)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=repository, text=True).strip()
    if revision != PITOME_REVISION:
        raise RuntimeError('PiToMe revision differs from the source-reviewed pin')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', PITOME_SOURCE],
                   cwd=repository, capture_output=True, check=True)
    spec = importlib.util.spec_from_file_location('_author_pitome_merge',
                                                  repository / PITOME_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pitome_backend(repository):
    module = author_pitome(repository)

    def matching(metric, count):
        if metric.ndim != 3 or metric.dtype != torch.float32 or metric.shape[0] != 1:
            raise ValueError('Expected FP32 [1,tokens,channels] matching features')
        tokens = metric.shape[1]
        if tokens == 0 or not torch.isfinite(metric).all() or (metric.norm(dim=-1) == 0).any():
            raise ValueError('Matching features must be nonempty, finite and nonzero')
        if type(count) is not int or not 0 <= count <= tokens // 4:
            raise ValueError('Protection study allows at most quarter-group removal')
        # Upstream returns a tuple for ratio==1 but a callable otherwise;
        # positive ratios rounding to zero removals also enter an empty max.
        if count == 0:
            return module.do_nothing, None
        ratio = 1 - count / tokens
        # The author computes floor(T-T*ratio). Round down one ULP only
        # when necessary, so odd groups retain the exact declared budget.
        if math.floor(tokens - tokens * ratio) < count:
            ratio = math.nextafter(ratio, -math.inf)
        if math.floor(tokens - tokens * ratio) != count:
            raise RuntimeError('Cannot express the exact upstream merge budget')
        merge = module.pitome_vision(metric, ratio=ratio, margin=.5,
                                     class_token=False, alpha=1.,
                                     use_bsm=False, use_bsm_pitome=False)
        return merge, None

    return SimpleNamespace(bipartite_soft_matching=matching,
                           merge_wavg=module.merge_wavg,
                           merge_source=module.merge_source)


@contextmanager
def use_pitome(repository):
    """Use author energy ranking/aggregation without importing its model stack."""
    with patch('modeling.region_constrained_tome.author_tome',
               return_value=pitome_backend(repository)):
        yield
