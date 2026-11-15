#!/usr/bin/env python3
"""Freeze a label-free manifest for the fully audited source-group captures.

This is a separate training bridge. It does not run inference, load ground-truth
mask pixels, promote a model, or consume partial/short captures. Run only after
all nine full reference/capture pairs and their independent audits complete.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import torch
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from scripts.audit_source_group_capture_v1 import verify_trajectory, verify_observation_coverage, verify_sample_file
from scripts.freeze_source_group_runtime_v1 import current_identity, validate_source_freeze
from scripts.memory_token_io import digest, validate_output
from scripts.run_source_group_capture_v1 import verify_short_gate
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture

SEQUENCES = ('seq2', 'seq3', 'seq4', 'seq5', 'seq12', 'seq13', 'seq14', 'seq9', 'seq15')
VALIDATION = {'seq9', 'seq15'}


def read(path):
    return json.loads(Path(path).read_text())


def read_unique_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate capture-map key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def label_hash_projection(path):
    """Extract only archived indices and label hashes; don't decode mask files.

    The historical metadata bytes also contain prior label-derived scores.
    Those fields are not parsed or used here, but this is development data and
    not a previously unseen validation set.
    """
    objects = re.findall(rb'\{[^{}]*\}', Path(path).read_bytes())
    rows = []
    for obj in objects:
        row = {}
        for key, pattern in (('index', rb'"index"\s*:\s*(\d+)'),
                             ('source_frame_index', rb'"source_frame_index"\s*:\s*(\d+)'),
                             ('label_sha256', rb'"label_sha256"\s*:\s*"([0-9a-f]{64})"')):
            hits = re.findall(pattern, obj)
            if len(hits) != 1:
                raise ValueError('Ambiguous archived frame identity or label hash')
            row[key] = hits[0].decode() if key == 'label_sha256' else int(hits[0])
        rows.append(row)
    if not rows:
        raise ValueError('No archived frame metadata')
    return rows


def resolve_capture_map(mapping):
    """Require an explicit, non-aliased reference/capture/audit for every video."""
    if set(mapping) != set(SEQUENCES):
        raise ValueError('Capture map must name every frozen sequence exactly once')
    result, used = {}, set()
    for seq in SEQUENCES:
        row = mapping[seq]
        if not isinstance(row, dict) or set(row) != {'reference', 'capture', 'audit'}:
            raise ValueError(f'{seq}: capture map requires reference, capture and audit paths')
        folders = tuple(Path(row[k]).resolve() for k in ('reference', 'capture', 'audit'))
        for folder in folders:
            validate_output(folder)
            if folder in used:
                raise ValueError('Capture map reuses or aliases a directory')
            used.add(folder)
        result[seq] = folders
    return result


def verify_capture_metadata(sequence, spec, reference, capture, audit, *, preflight_sha, freeze_sha, gate_sha, identity):
    """Validate complete audited coverage without loading images or labels."""
    if (not reference.get('passed') or not capture.get('passed') or not audit.get('passed')
            or reference.get('mode') != 'reference' or capture.get('mode') != 'capture'
            or any(r.get('sequence') != sequence for r in (reference, capture, audit))
            or reference.get('identity') != identity or capture.get('identity') != identity
            or audit.get('identity') != identity or identity.get('preflight_sha256') != preflight_sha
            or identity.get('source_freeze_sha256') != freeze_sha
            or reference.get('short_gate_sha256') != gate_sha or capture.get('short_gate_sha256') != gate_sha
            or any(r.get('limit') != 0 or r.get('full_sequence') is not True for r in (reference, capture, audit))
            or audit.get('auditor_sha256') != digest(ROOT/'scripts/audit_source_group_capture_v1.py')
            or any(r.get('label_guard', {}).get('open_attempts') != 0 for r in (reference, capture, audit))):
        raise ValueError(f'{sequence}: incomplete or mismatched full-capture evidence')
    if verify_trajectory(reference, capture, spec) != (0, spec['targets']):
        raise ValueError(f'{sequence}: trajectory coverage differs')
    verify_observation_coverage(capture['observations'], spec['targets'])
    if audit.get('targets') != spec['targets'] or audit.get('frames') != spec['frames']-spec['prompt_frame']:
        raise ValueError(f'{sequence}: audit coverage differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight', 'source-freeze', 'short-gate', 'reference-capture', 'checkpoint', 'capture-map', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    label_guard = install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl')
    torch.set_num_threads(2)
    pre_path = a.preflight/'manifest.json'; pre = read(pre_path)
    freeze_path = a.source_freeze/'source_freeze.json'; freeze = read(freeze_path)
    validate_source_freeze(freeze, read(a.source_freeze/'completion.json'), digest(freeze_path),
                           current_identity(a.preflight, a.reference_capture, a.checkpoint))
    gate = read(a.short_gate)
    verify_short_gate(gate, digest(pre_path), digest(freeze_path),
                      digest(ROOT/'scripts/run_source_group_capture_v1.py'),
                      digest(ROOT/'scripts/audit_source_group_capture_v1.py'), gate['identity']['runtime_environment'])
    if set(pre['sequences']) != set(SEQUENCES):
        raise ValueError('Unexpected source-group cohort')
    for seq in SEQUENCES:
        expected_split = 'validation' if seq in VALIDATION else 'train'
        if pre['sequences'][seq]['split'] != expected_split:
            raise ValueError('Frozen sequence split differs')
    paths = resolve_capture_map(read_unique_json(a.capture_map))
    missing = [f'{seq}/{kind}' for seq, folders in paths.items()
               for kind, folder in zip(('reference', 'capture', 'audit'), folders, strict=True)
               if not (folder/('verification.json' if kind == 'audit' else 'receipt.json')).is_file()]
    if missing:
        raise ValueError('Full captures or audits still missing: ' + ', '.join(missing))
    videos = load_manifest(ROOT/'data/polypgen_sequence.jsonl', 'dev')
    common = gate['identity']
    evidence, samples = {}, []
    for seq in SEQUENCES:
        spec = pre['sequences'][seq]
        rdir, cdir, adir = paths[seq]
        rp, cp, ap = rdir/'receipt.json', cdir/'receipt.json', adir/'verification.json'
        ref, cap, audit = read(rp), read(cp), read(ap)
        verify_capture_metadata(seq, spec, ref, cap, audit, preflight_sha=digest(pre_path),
                                freeze_sha=digest(freeze_path), gate_sha=digest(a.short_gate), identity=common)
        if (Path(audit['reference_receipt_path']).resolve() != rp.resolve()
                or Path(audit['capture_receipt_path']).resolve() != cp.resolve()
                or audit['reference_receipt_sha256'] != digest(rp) or audit['capture_receipt_sha256'] != digest(cp)
                or audit['label_guard']['blocked_paths'] != label_guard['blocked_paths']
                or any(r['label_guard']['blocked_paths'] != label_guard['blocked_paths'] for r in (ref, cap))):
            raise ValueError(f'{seq}: audit receipt path, hash or label guard differs')
        archive = Path(spec['source'])/seq
        if (digest(archive/'frames.json') != spec['archived_frames_sha256']
                or digest(archive/'prompt.json') != spec['archived_prompt_sha256']
                or digest(archive/'prompt_refined.png') != spec['archived_prompt_mask_sha256']):
            raise ValueError(f'{seq}: archived prompt or metadata changed')
        prompt = np.asarray(Image.open(archive/'prompt_refined.png').convert('L')) > 0
        for folder, rec in ((rdir, ref), (cdir, cap)):
            if digest(folder/'conditioning_mask.png') != rec['conditioning_sha256']:
                raise ValueError(f'{seq}: conditioning mask changed')
            with Image.open(folder/'conditioning_mask.png') as img:
                if not np.array_equal(np.asarray(img.convert('L')) > 0, prompt):
                    raise ValueError(f'{seq}: conditioning differs from frozen prompt')
        frames, metadata = videos[seq], label_hash_projection(archive/'frames.json')
        if len(frames) != spec['frames'] or len(metadata) != len(frames):
            raise ValueError(f'{seq}: manifest frame coverage differs')
        for i, (frame, row) in enumerate(zip(frames, metadata, strict=True)):
            if (row['index'] != i or row['source_frame_index'] != frame.frame_index
                    or digest(frame.image_path) != spec['image_sha256'][i]):
                raise ValueError(f'{seq}: source frame changed')
        for row in ref['rows']:
            for folder in (rdir, cdir):
                if digest(folder/f"{row['frame']:06d}.png") != row['mask_sha256']:
                    raise ValueError(f'{seq}: frozen trajectory changed')
        for obs in cap['observations']:
            i = obs['frame']; frame = frames[i]
            feature, snapshot = Path(obs['feature_file']), Path(obs['snapshot_file'])
            verify_sample_file(feature, cdir, obs['feature_sha256'])
            verify_sample_file(snapshot, cdir, obs['snapshot_sha256'])
            item = {'split': spec['split'], 'sequence': seq, 'frame': i, 'source_frame_index': frame.frame_index,
                    'image': str(frame.image_path), 'image_sha256': spec['image_sha256'][i],
                    'label': str(frame.mask_path), 'label_sha256': metadata[i]['label_sha256'],
                    'feature': str(feature), 'feature_sha256': obs['feature_sha256'],
                    'snapshot': str(snapshot), 'snapshot_sha256': obs['snapshot_sha256'], 'capture_device': 'mps'}
            raw_feature = torch.load(feature, map_location='cpu', weights_only=True)
            raw_snapshot = torch.load(snapshot, map_location='cpu', weights_only=True)
            raw_tensors = (raw_feature['feature'], raw_snapshot['decoder_inputs']['backbone_features'],
                           raw_snapshot['decoder_outputs'][3])
            if any(x.dtype != torch.float32 for x in raw_tensors):
                raise ValueError(f'{seq}/{i}: saved tensor dtype differs')
            with Image.open(frame.image_path) as img:
                if (raw_snapshot['video_width'], raw_snapshot['video_height']) != img.size:
                    raise ValueError(f'{seq}/{i}: captured video geometry differs')
            del raw_feature, raw_snapshot, raw_tensors
            # This validates tensor identity, geometry, finite values and native
            # low/high-resolution resize without opening any label file.
            read_capture(item)
            samples.append(item)
        evidence[seq] = {'reference_receipt_path': str(rp.resolve()), 'capture_receipt_path': str(cp.resolve()),
                         'audit_path': str(ap.resolve()), 'reference_receipt_sha256': digest(rp),
                         'capture_receipt_sha256': digest(cp), 'audit_sha256': digest(ap), 'targets': spec['targets'],
                         'pre_prompt_frames': spec['prompt_frame'], 'source_frames': spec['frames']}
    if (len(samples) != 214 or len({(x['sequence'], x['frame']) for x in samples}) != 214
            or sum(x['split'] == 'train' for x in samples) != 164
            or sum(x['split'] == 'validation' for x in samples) != 50):
        raise ValueError('Training/validation sample coverage differs')
    pipeline = {'checkpoint_sha256': digest(a.checkpoint), 'source_freeze_sha256': digest(freeze_path),
                'short_gate_sha256': digest(a.short_gate), 'model_state_sha256': common['model_state_sha256'],
                'runtime_environment': common['runtime_environment'], 'observer_sha256': common['observer_sha256'],
                'historical_cpu_feature_pipeline_sha256': common['cpu_feature_pipeline_sha256'],
                'capture_device': 'mps'}
    result = {'passed': True, 'scope': 'Exploratory inspected videos; no patient-independence claim.',
              'preflight_path': str(pre_path.resolve()), 'source_freeze_path': str(freeze_path.resolve()),
              'capture_map_path': str(a.capture_map.resolve()), 'capture_map_sha256': digest(a.capture_map),
              'short_gate_path': str(a.short_gate.resolve()), 'checkpoint_path': str(a.checkpoint.resolve()),
              'reference_capture_path': str(a.reference_capture.resolve()),
              'preflight_sha256': digest(pre_path), 'source_freeze_sha256': digest(freeze_path),
              'short_gate_sha256': digest(a.short_gate), 'checkpoint_sha256': digest(a.checkpoint),
              'feature_pipeline': pipeline, 'feature_pipeline_sha256': identity_hash(pipeline),
              'sample_count': len(samples), 'sequences': evidence, 'samples': samples,
              'label_guard': dict(label_guard), 'runner_sha256': digest(Path(__file__)),
              'limitations': ['Historical frame metadata bytes contain label-derived scores; only label hashes and frame identities are projected.',
                              'Validation sequences were previously inspected and are exploratory, not an independent holdout.',
                              'Fixed-read capture audits do not establish a live-refiner video rollout.']}
    out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    path = out/'manifest.json'; path.write_text(json.dumps(result, indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed': True, 'manifest_sha256': digest(path)}, indent=2)+'\n')
    print(json.dumps({'passed': True, 'train': 164, 'validation': 50, 'mask_open_attempts': label_guard['open_attempts']}))


if __name__ == '__main__':
    main()
