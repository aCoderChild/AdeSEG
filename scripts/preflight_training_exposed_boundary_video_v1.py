#!/usr/bin/env python3
"""Read frozen automatic prompts for the nine prior training/validation videos.

The initial checkpoint has already seen these sequences during fitting or
selection. This loader is label-free and cannot make them held-out cases.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.video import load_manifest
from scripts.preflight_author_trainval_v1 import trainval_inputs
from scripts.memory_token_io import digest
from scripts.preflight_remaining_boundary_video_v1 import (
    CHECKPOINT_SHA, DATA, DETECTOR_SHA, EXPECTED_TRAIN, EXPECTED_VALIDATION,
    checked_training_preflight, image_metadata_only, read, validate_prompt,
)

SEQUENCES = ('seq2', 'seq3', 'seq4', 'seq5', 'seq9', 'seq12', 'seq13', 'seq14', 'seq15')


def load_training_exposed_preflight(preflight, checkpoint, detector):
    if digest(checkpoint) != CHECKPOINT_SHA or digest(detector) != DETECTOR_SHA:
        raise ValueError('Backbone or archived detector differs')
    data = checked_training_preflight(preflight)
    if (data.get('passed') is not True or data.get('manifest_sha256') != digest(DATA)
            or data.get('checkpoint_sha256') != CHECKPOINT_SHA
            or set(data['sequences']) != EXPECTED_TRAIN | EXPECTED_VALIDATION):
        raise ValueError('Frozen source-group identity differs')
    return data


def training_exposed_inputs(sequence, preflight, checkpoint, detector):
    if sequence not in SEQUENCES:
        raise ValueError('Sequence not in prior training/validation cohort')
    data = load_training_exposed_preflight(preflight, checkpoint, detector)
    item = data['sequences'][sequence]
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}
    frames = videos[sequence]
    if len(frames) != item['frames'] or len(item['image_sha256']) != len(frames):
        raise ValueError('Frozen input coverage differs')
    folder = (Path(item['source']) / sequence).resolve(strict=True)
    prompt_file, mask_file, frame_file = (folder / 'prompt.json', folder / 'prompt_refined.png', folder / 'frames.json')
    for name, file in (('archived_prompt_sha256', prompt_file),
                       ('archived_prompt_mask_sha256', mask_file),
                       ('archived_frames_sha256', frame_file)):
        if digest(file) != item[name]:
            raise ValueError(f'{sequence}: archived input changed: {name}')
    indices, archived_hashes = image_metadata_only(frame_file)
    if len(indices) != len(frames) or archived_hashes != item['image_sha256']:
        raise ValueError('Archived image identities differ')
    for i, frame in enumerate(frames):
        if frame.frame_index != indices[i] or digest(frame.image_path) != archived_hashes[i]:
            raise ValueError(f'{sequence}: frame source differs at {i}')
    prompt = read(prompt_file)
    start = prompt.get('first_detection_index')
    if start != item['prompt_frame'] or not 0 <= start < len(frames) - 1:
        raise ValueError('Archived automatic initialization differs')
    if prompt.get('refined_mask_sha256') != digest(mask_file):
        raise ValueError('Archived prompt mask hash differs')
    with Image.open(frames[start].image_path) as image:
        size = image.size
    mask = validate_prompt(prompt, mask_file, size)
    other_frames, other_start, other_mask, other_provenance = trainval_inputs(
        sequence, checkpoint, detector, preflight)
    if (other_start != start or len(other_frames) != len(frames) or not np.array_equal(other_mask, mask)
            or other_provenance['prompt_metadata_sha256'] != digest(prompt_file)
            or other_provenance['prompt_mask_sha256'] != digest(mask_file)):
        raise ValueError('Independent automatic-prompt lineage check differs')
    provenance = {'prompt_file': str(mask_file), 'prompt_mask_sha256': digest(mask_file),
                  'prompt_metadata_sha256': digest(prompt_file),
                  'reference_receipt_sha256': item['archived_completion_sha256'],
                  'preflight_sha256': digest(Path(preflight) / 'manifest.json'),
                  'roi_xyxy': [0, 0, *size]}
    return frames, prompt, digest(prompt_file), {'kind': 'full_source_frame'}, mask, provenance
