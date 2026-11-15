#!/usr/bin/env python3
"""Add two retrospectively selected seq4 missed-target training reads.

The two new samples were selected using training labels and archived predictions.
Validation remains previously inspected and is not an independent test set.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output
from scripts.train_multilevel_refiner_seq4_empty_v2 import check_manifest, read_capture


def read(path):
    return json.loads(path.read_text())


def validate_samples(samples):
    keys = [(x['sequence'], x['frame']) for x in samples]
    train = [x for x in samples if x['split'] == 'train']
    val = [x for x in samples if x['split'] == 'validation']
    if len(keys) != 29 or len(set(keys)) != 29 or len(train) != 19 or len(val) != 10:
        raise ValueError('Expected 19 train and 10 validation distinct samples')
    if {x['sequence'] for x in train} != {'seq2', 'seq3', 'seq4'} or {x['sequence'] for x in val} != {'seq9', 'seq10'}:
        raise ValueError('Unexpected sequence split')
    if any(sum(x['sequence'] == seq for x in train) != 5 for seq in ('seq2', 'seq3')):
        raise ValueError('Unexpected seq2/seq3 training coverage')
    if {x['frame'] for x in train if x['sequence'] == 'seq4'} != {10, 17, 18, 19, 21, 24, 28, 38, 47}:
        raise ValueError('Unexpected seq4 training frames')
    if any(sum(x['sequence'] == seq for x in val) != 5 for seq in ('seq9', 'seq10')):
        raise ValueError('Unexpected validation coverage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous-preflight', 'split-preflight', 'extra-capture', 'prior-capture', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    original, original_sha = check_manifest(a.previous_preflight)
    split_path = a.split_preflight / 'manifest.json'
    split = read(split_path)
    split_done = read(a.split_preflight / 'completion.json')
    if not split_done['passed'] or split_done['manifest_sha256'] != digest(split_path) or original['split_sha256'] != digest(split_path):
        raise ValueError('Sequence split changed')
    receipt_path = a.extra_capture / 'extra_capture_receipt.json'
    extra = read(receipt_path)
    prior_feature = a.prior_capture / 'channel_feature_receipt.json'
    if (not extra['passed'] or extra['sequence'] != 'seq4' or extra['targets'] != [19, 21, 24]
            or extra['feature_pipeline_sha256'] != original['feature_pipeline_sha256']
            or extra['prior_feature_receipt_sha256'] != digest(prior_feature)
            or extra['prior_utility_receipt_sha256'] != digest(a.prior_capture / 'utility_capture_receipt.json')
            or extra['records_sha256'] != digest(a.extra_capture / 'records.json')
            or extra['records_sha256'] != digest(a.prior_capture / 'records.json')
            or extra['runner_sha256'] != digest(ROOT / 'scripts/capture_seq4_missed_refiner_reads.py')):
        raise ValueError('Hard-case capture provenance differs')
    source = Path(split['sequences']['seq4']['source']) / 'seq4' / 'frames.json'
    if digest(source) != split['sequences']['seq4']['frames_metadata_sha256']:
        raise ValueError('Source-frame metadata changed')
    rows = read(source)
    frozen_records = read(a.extra_capture / 'records.json')
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    samples = list(original['samples'])
    for item in samples:
        if any(digest(item[key]) != item[key + '_sha256'] for key in ('image', 'label', 'feature', 'snapshot')):
            raise ValueError('Previous sample changed')
    for i in (21, 24):
        frame = videos['seq4'][i]
        entry = next(x for x in extra['frames'] if x['frame'] == i)
        feature = a.extra_capture / 'features' / f'{i:06d}.pt'
        snapshot = a.extra_capture / 'snapshots' / f'{i:06d}.pt'
        if (digest(frame.image_path) != rows[i]['image_sha256'] or digest(frame.mask_path) != rows[i]['label_sha256']
                or digest(feature) != entry['feature_sha256'] or digest(snapshot) != entry['snapshot_sha256']):
            raise ValueError('Hard-case image, label or features changed')
        target = load_label_mask(frame.mask_path, [1]) == 1
        if not target.any():
            raise ValueError('Hard case has no labeled foreground')
        prediction_path = a.extra_capture / f'{i:06d}.png'
        if digest(prediction_path) != frozen_records['mask_sha256'][i]:
            raise ValueError('Hard-case archived mask changed')
        prediction = np.asarray(Image.open(prediction_path).convert('L')) > 0
        if prediction.shape != target.shape or (prediction & target).any():
            raise ValueError('Hard case is not a verified zero-overlap prediction')
        item = {'split': 'train', 'sequence': 'seq4', 'frame': i,
                'image': str(frame.image_path), 'image_sha256': rows[i]['image_sha256'],
                'label': str(frame.mask_path), 'label_sha256': rows[i]['label_sha256'],
                'feature': str(feature), 'feature_sha256': entry['feature_sha256'],
                'snapshot': str(snapshot), 'snapshot_sha256': entry['snapshot_sha256']}
        read_capture(item)
        if i == 24:
            snap = torch.load(snapshot, map_location='cpu', weights_only=True)
            if not torch.all(snap['decoder_outputs'][3] == -1024):
                raise ValueError('Expected native no-object suppression at frame 24')
            if not torch.all(snap['decoder_outputs'][6] < 0):
                raise ValueError('Expected negative native object score at frame 24')
        samples.append(item)
    validate_samples(samples)
    result = {'passed': True, 'samples': samples,
              'checkpoint_sha256': original['checkpoint_sha256'],
              'feature_pipeline_sha256': original['feature_pipeline_sha256'],
              'split_sha256': original['split_sha256'],
              'previous_manifest_path': str((a.previous_preflight / 'manifest.json').resolve()),
              'previous_manifest_sha256': original_sha,
              'extra_receipt_path': str(receipt_path.resolve()), 'extra_receipt_sha256': digest(receipt_path),
              'source_sha256': digest(Path(__file__)),
              'selection': 'Seq4 frames 21/24 were selected after inspecting labels and predictions as training-only misses; frame 19 is not duplicated.',
              'scope': 'Exploratory 19 training/10 previously inspected validation reads; patient independence unknown.'}
    out = validate_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'manifest_sha256': digest(out / 'manifest.json')}, indent=2) + '\n')
    print(json.dumps({'passed': True, 'train': 19, 'validation': 10}))


if __name__ == '__main__':
    main()
