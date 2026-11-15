#!/usr/bin/env python3
"""Replay seq4's verified recent token-window trajectory and capture 19/21/24.

Frames 21 and 24 were chosen after label inspection as missed-polyp development
examples; frame 19 overlaps an earlier capture for an exact observer check.
This is data capture, not efficacy evaluation or independent validation.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.lean_channel_feature_capture import capture_channel_features
from modeling.lean_memory_capture import capture_memory_read
from modeling.lean_token_window_storage import TokenWindowRuntime
from scripts.memory_token_io import digest
from scripts.run_lean_offline_memory_ablation import equal
from scripts.run_lean_token_window import FORK, PATCHED
import scripts.run_lean_indexed_spatial_rope as runner

TARGETS = {19, 21, 24}
TRACE_FILES = ('records.json', 'positions.json', 'selections.json', 'memory_events.json')


def read(path):
    return json.loads(path.read_text())


def require_reference(reference, checkpoint):
    feature = read(reference / 'channel_feature_receipt.json')
    utility = read(reference / 'utility_capture_receipt.json')
    done = read(reference / 'completion.json')
    if (not feature['passed'] or not utility['passed'] or not done['passed']
            or feature['sequence'] != 'seq4' or utility['sequence'] != 'seq4'
            or feature['checkpoint_sha256'] != digest(checkpoint)
            or feature['records_sha256'] != digest(reference / 'records.json')
            or done['records_sha256'] != digest(reference / 'records.json')
            or feature['events_sha256'] != digest(reference / 'token_window_events.json')
            or utility['feature_receipt_sha256'] != digest(reference / 'channel_feature_receipt.json')):
        raise ValueError('Prior capture or model identity changed')
    raw = feature['pipeline']['raw_identity']
    local = {'runner_sha256': ROOT / 'scripts/run_lean_token_window.py',
             'protocol_sha256': ROOT / 'protocols/lean_token_window_candidate_v1.json',
             'policy_sha256': ROOT / 'modeling/token_memory_window.py',
             'runtime_sha256': ROOT / 'modeling/lean_token_window_storage.py',
             'observer_sha256': ROOT / 'modeling/lean_token_storage_observer.py'}
    if any(raw[key] != digest(path) for key, path in local.items()):
        raise ValueError('Token-window source identity changed')
    if any(raw['patch_sha256'][key] != digest(FORK / key) for key in PATCHED):
        raise ValueError('Patched Lean source identity changed')
    feature19 = next(x for x in feature['features'] if x['frame'] == 19)
    snapshot19 = next(x for x in utility['snapshots'] if x['frame'] == 19)
    if (feature19['file_sha256'] != digest(reference / 'features/000019.pt')
            or snapshot19['sha256'] != digest(reference / 'snapshots/000019.pt')):
        raise ValueError('Prior overlapping snapshot changed')
    return feature


def verify_match(reference, output, runtime, feature_events, read_events):
    if {x['frame'] for x in feature_events} != TARGETS or {x['frame'] for x in read_events} != TARGETS:
        raise ValueError('Requested feature or decoder reads missing')
    if any(digest(reference / name) != digest(output / name) for name in TRACE_FILES):
        raise ValueError('Capture changed the prior trajectory trace')
    if runtime.events != read(reference / 'token_window_events.json'):
        raise ValueError('Capture changed the prior token-window events')
    prior = read(reference / 'records.json')
    if prior['frames'] != 48:
        raise ValueError('Unexpected seq4 frame count')
    for i in range(prior['frames']):
        if (digest(reference / f'{i:06d}.png') != prior['mask_sha256'][i]
                or digest(output / f'{i:06d}.png') != prior['mask_sha256'][i]):
            raise ValueError(f'Capture changed frame {i}')
    for path in ('features/000019.pt', 'snapshots/000019.pt'):
        old = torch.load(reference / path, map_location='cpu', weights_only=True)
        new = torch.load(output / path, map_location='cpu', weights_only=True)
        if not equal(old, new):
            raise ValueError('Overlapping capture differs: ' + path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'merged', 'reference', 'cpu-verification',
                 'observed-reference', 'gate-a', 'gate-b', 'prior-capture', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    prior = require_reference(args.prior_capture, args.checkpoint)
    observations, reads, positions = [], [], []
    runtime = None
    build, rope = runner.build_author_predictor, runner.indexed_spatial_rope

    def build_fork(config, _repo):
        nonlocal runtime
        sys.path.insert(0, str(FORK))
        model = build(config, FORK)
        import sam2
        if (not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve())
                or model.num_maskmem != 7 or model.mem_keep_ratio != .25
                or model.insurance_max_frames != 3 or not model.enable_insurance_bank):
            raise ValueError('Unexpected recent-window model configuration')
        model.ema_history_replace_old_tail_in_attn = False
        model.ema_history_resident_prune = False
        runtime = TokenWindowRuntime(16384, 3072)
        model.token_window_runtime = runtime
        return model

    @contextmanager
    def capture(model, records):
        with rope(model, records), capture_channel_features(model, TARGETS, args.output, observations), \
                capture_memory_read(model, TARGETS, args.output, reads):
            yield

    old_argv = sys.argv
    runner.build_author_predictor, runner.indexed_spatial_rope = build_fork, capture
    try:
        sys.argv = ['run_lean_indexed_spatial_rope.py', '--mode', 'candidate', '--sequence', 'seq4']
        for key in ('checkpoint', 'merged', 'reference', 'cpu_verification',
                    'observed_reference', 'gate_a', 'gate_b', 'output'):
            sys.argv.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
        runner.main()
    finally:
        runner.build_author_predictor, runner.indexed_spatial_rope = build, rope
        sys.argv = old_argv
    if runtime is None:
        raise ValueError('Token runtime not installed')
    verify_match(args.prior_capture, args.output, runtime, observations, reads)
    result = {'passed': True, 'sequence': 'seq4', 'targets': sorted(TARGETS),
              'selection': '21 and 24 selected as missed-polyp training examples after label inspection; 19 is an overlap control',
              'scope': 'development-only capture, not independent evaluation or model efficacy',
              'feature_pipeline_sha256': prior['feature_pipeline_sha256'],
              'prior_feature_receipt_sha256': digest(args.prior_capture / 'channel_feature_receipt.json'),
              'prior_utility_receipt_sha256': digest(args.prior_capture / 'utility_capture_receipt.json'),
              'records_sha256': digest(args.output / 'records.json'),
              'runner_sha256': digest(Path(__file__)),
              'frames': [{'frame': i, 'feature_sha256': digest(args.output / 'features' / f'{i:06d}.pt'),
                          'snapshot_sha256': digest(args.output / 'snapshots' / f'{i:06d}.pt')} for i in sorted(TARGETS)]}
    (args.output / 'extra_capture_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'sequence': 'seq4', 'targets': sorted(TARGETS)}))


if __name__ == '__main__':
    main()
