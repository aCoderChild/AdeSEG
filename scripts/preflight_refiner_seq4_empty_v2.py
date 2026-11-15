#!/usr/bin/env python3
"""Bind two label-selected seq4 empty reads to the earlier fixed-read pilot."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output
from scripts.train_multilevel_refiner_fixed_read_pilot import check_manifest, read_capture


def read(path):
    return json.loads(path.read_text())


def validate_samples(samples):
    keys = [(x['sequence'], x['frame']) for x in samples]
    if len(keys) != 27 or len(set(keys)) != 27:
        raise ValueError('Expected 27 unique sequence/frame samples')
    train = [x for x in samples if x['split'] == 'train']
    validation = [x for x in samples if x['split'] == 'validation']
    if len(train) != 17 or len(validation) != 10:
        raise ValueError('Expected 17 train and 10 validation samples')
    if {x['sequence'] for x in train} != {'seq2', 'seq3', 'seq4'} or {x['sequence'] for x in validation} != {'seq9', 'seq10'}:
        raise ValueError('Unexpected sequence allocation')
    for seq, count in [('seq2', 5), ('seq3', 5), ('seq4', 7)]:
        if sum(x['sequence'] == seq for x in train) != count:
            raise ValueError('Unexpected per-sequence training coverage')
    if any(sum(x['sequence'] == seq for x in validation) != 5 for seq in ('seq9', 'seq10')):
        raise ValueError('Unexpected per-sequence validation coverage')
    if any(x['split'] != 'train' for x in samples if x['sequence'] == 'seq4' and x['frame'] in (17, 18)):
        raise ValueError('Label-selected examples must remain training-only')
    if not {('seq4', 17), ('seq4', 18)}.issubset(set(keys)):
        raise ValueError('Missing label-selected seq4 examples')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original-preflight', 'split-preflight', 'extra-capture', 'prior-capture', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    original, original_sha = check_manifest(a.original_preflight)
    split = read(a.split_preflight / 'manifest.json')
    split_done = read(a.split_preflight / 'completion.json')
    if not split_done['passed'] or original['split_sha256'] != digest(a.split_preflight / 'manifest.json') or split_done['manifest_sha256'] != original['split_sha256']:
        raise ValueError('Earlier sequence split changed')
    extra = read(a.extra_capture / 'extra_capture_receipt.json')
    prior = read(a.prior_capture / 'channel_feature_receipt.json')
    if (not extra['passed'] or extra['sequence'] != 'seq4' or extra['targets'] != [17, 18, 19]
            or extra['feature_pipeline_sha256'] != original['feature_pipeline_sha256']
            or extra['prior_feature_receipt_sha256'] != digest(a.prior_capture / 'channel_feature_receipt.json')
            or extra['prior_utility_receipt_sha256'] != digest(a.prior_capture / 'utility_capture_receipt.json')
            or extra['records_sha256'] != digest(a.extra_capture / 'records.json')
            or extra['records_sha256'] != digest(a.prior_capture / 'records.json')
            or extra['runner_sha256'] != digest(ROOT / 'scripts/capture_seq4_empty_refiner_reads.py')
            or prior['checkpoint_sha256'] != original['checkpoint_sha256']):
        raise ValueError('Additional capture provenance differs')
    source = Path(split['sequences']['seq4']['source']) / 'seq4' / 'frames.json'
    if digest(source) != split['sequences']['seq4']['frames_metadata_sha256']:
        raise ValueError('Archived frame source changed')
    rows = read(source)
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    samples = list(original['samples'])
    for old in samples:
        for key in ('image', 'label', 'feature', 'snapshot'):
            if digest(old[key]) != old[key + '_sha256']:
                raise ValueError('Prior sample changed: ' + key)
    for i in (17, 18):
        frame = videos['seq4'][i]
        entry = next(x for x in extra['frames'] if x['frame'] == i)
        feature = a.extra_capture / 'features' / f'{i:06d}.pt'
        snapshot = a.extra_capture / 'snapshots' / f'{i:06d}.pt'
        if (digest(frame.image_path) != rows[i]['image_sha256'] or digest(frame.mask_path) != rows[i]['label_sha256']
                or digest(feature) != entry['feature_sha256'] or digest(snapshot) != entry['snapshot_sha256']):
            raise ValueError('Additional feature or source changed')
        if (load_label_mask(frame.mask_path, [1]) == 1).any():
            raise ValueError('Selected training example is not labeled empty')
        item = {'split': 'train', 'sequence': 'seq4', 'frame': i,
                'image': str(frame.image_path), 'image_sha256': rows[i]['image_sha256'],
                'label': str(frame.mask_path), 'label_sha256': rows[i]['label_sha256'],
                'feature': str(feature), 'feature_sha256': entry['feature_sha256'],
                'snapshot': str(snapshot), 'snapshot_sha256': entry['snapshot_sha256']}
        read_capture(item)
        samples.append(item)
    validate_samples(samples)
    result = {'passed': True, 'samples': samples,
              'checkpoint_sha256': original['checkpoint_sha256'],
              'feature_pipeline_sha256': original['feature_pipeline_sha256'],
              'split_sha256': original['split_sha256'], 'original_manifest_sha256': original_sha,
              'original_manifest_path': str((a.original_preflight / 'manifest.json').resolve()),
              'extra_receipt_sha256': digest(a.extra_capture / 'extra_capture_receipt.json'),
              'extra_receipt_path': str((a.extra_capture / 'extra_capture_receipt.json').resolve()),
              'source_sha256': digest(Path(__file__)),
              'selection': 'Seq4 frames 17/18 were selected after inspecting labels as empty training examples; frame 19 is not duplicated.',
              'scope': 'Exploratory development: 17 training and 10 previously inspected validation reads; patient independence unknown.'}
    out = validate_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'manifest_sha256': digest(out / 'manifest.json')}, indent=2) + '\n')
    print(json.dumps({'passed': True, 'train': 17, 'validation': 10}))


if __name__ == '__main__':
    main()
