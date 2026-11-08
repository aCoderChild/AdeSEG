"""Sample manifest clips with the pinned SAM2 author sampler."""
import ast
import random
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import List

import numpy as np

from datasets.io import load_label_mask
from datasets.instance_masks import load_instance_mask
from modeling.region_tokens import SAM2_REVISION, _verified_source


@lru_cache(maxsize=1)
def _sampler_class():
    path, source = _verified_source('sam2', SAM2_REVISION, 'training/dataset/vos_sampler.py')
    names = {'SampledFramesAndObjects', 'VOSSampler', 'RandomUniformSampler'}
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = {'random': random, 'dataclass': dataclass, 'List': List, 'MAX_RETRIES': 1000,
                 'LazySegments': type('LazySegments', (), {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['RandomUniformSampler']


class ManifestSegments:
    def __init__(self, frames, label_ids=(1,)):
        self.paths = {frame.frame_index: frame.mask_path for frame in frames}
        self.label_ids = tuple(label_ids)

    def load(self, frame_id):
        if self.label_ids == (1,):
            labels = load_label_mask(self.paths[frame_id], self.label_ids)
        else:
            labels = load_instance_mask(self.paths[frame_id])
            if set(np.unique(labels).tolist()) - {0, *self.label_ids}:
                raise ValueError('Mask contains undeclared instance IDs')
        return {object_id: labels == object_id for object_id in self.label_ids}


def sample_clip(frames, length, seed):
    """Return author-sampled manifest frames without modifying global RNG state."""
    return sample_object_clip(frames, length, seed, label_ids=(1,), max_objects=1)[0]


def sample_object_clip(frames, length, seed, *, label_ids, max_objects):
    """Return author-sampled frames and their selected persistent instance IDs."""
    label_ids = tuple(label_ids)
    if (not label_ids or len(set(label_ids)) != len(label_ids)
            or any(not isinstance(i, int) or isinstance(i, bool) or i <= 0 for i in label_ids)
            or max_objects < 1):
        raise ValueError('Expected unique positive instance IDs and a positive object limit')
    if length < 2 or length > len(frames):
        raise ValueError('Clip length must be between 2 and the sequence length')
    wrapped = [SimpleNamespace(frame_idx=frame.frame_index, original=frame) for frame in frames]
    video = SimpleNamespace(frames=wrapped, video_name=frames[0].video_id)
    sampler = _sampler_class()(num_frames=length, max_num_objects=max_objects, reverse_time_prob=0.0)
    state = random.getstate()
    try:
        random.seed(seed)
        result = sampler.sample(video, ManifestSegments(frames, label_ids))
    finally:
        random.setstate(state)
    return [frame.original for frame in result.frames], list(result.object_ids)
