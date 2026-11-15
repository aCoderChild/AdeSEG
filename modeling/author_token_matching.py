"""Pinned upstream matching backends for controlled SAM2 memory experiments."""
from contextlib import contextmanager
from functools import lru_cache
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import torch
from modeling.region_constrained_tome import author_tome

ROOT = Path(__file__).resolve().parents[1]
ATC_REVISION = '227c94b584df323b94dd79e8ddbb88b1587eeb8e'
ATC_SOURCE = 'SSLBackbone_setup/token_reduc/merge.py'


@lru_cache(maxsize=1)
def author_atc():
    repo = ROOT / 'external/ATC'
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if actual != ATC_REVISION:
        raise RuntimeError('ATC revision differs from the protocol')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', ATC_SOURCE],
                   cwd=repo, check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location('_author_atc_merge', repo / ATC_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matching_backend(strategy):
    if strategy == 'tome':
        return author_tome()
    if strategy != 'atc':
        raise ValueError('Expected tome or atc')
    module = author_atc()

    def matching(metric, count):
        if metric.dtype != torch.float32 or metric.ndim != 3:
            raise ValueError('ATC integration requires FP32 [batch,tokens,channels]')
        if not torch.isfinite(metric).all() or (metric.norm(dim=-1) == 0).any():
            raise ValueError('ATC matching features must be finite and nonzero')
        if count < 0 or count > metric.shape[1] // 2:
            raise ValueError('Matched comparison permits removal of at most half the tokens')
        return module.agglomerative_clustering(metric, count, linkage='average')

    # Reuse author weighted sum/mass aggregation. Calling ATC's raw mean mode
    # directly would include its zero-initialized destinations in the mean.
    # Source maps also avoid the upstream unmerge closure's metric-width
    # assumption when value vectors have a different channel dimension.
    return SimpleNamespace(bipartite_soft_matching=matching,
                           merge_wavg=module.merge_wavg, merge_source=module.merge_source)


@contextmanager
def use_matching(strategy):
    """Sequential/process-local swap; existing Run155 merger remains untouched."""
    backend = matching_backend(strategy)
    with patch('modeling.region_constrained_tome.author_tome', return_value=backend):
        yield
