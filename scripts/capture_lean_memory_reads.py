#!/usr/bin/env python3
"""Capture selected reads during a replay and reject any trajectory difference."""
import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.lean_memory_capture import capture_memory_read
from scripts.memory_token_io import digest
import scripts.run_lean_indexed_spatial_rope as runner

PROTOCOL = ROOT / 'protocols/lean_memory_offline_ablation_v1.json'

def read(path):
    return json.loads(path.read_text())

def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'merged', 'reference', 'indexed-reference', 'cpu-verification', 'observed-reference', 'indexed-gate-a', 'indexed-gate-b', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--label-free-gate', action='store_true')
    a = p.parse_args()
    if (a.label_free_gate and a.sequence != 'seq3') or (not a.label_free_gate and a.sequence not in read(PROTOCOL)['cohort']):
        raise ValueError('Out-of-scope sequence')
    ref = read(a.indexed_reference / 'records.json')
    done = read(a.indexed_reference / 'completion.json')
    if not done['passed'] or digest(a.indexed_reference / 'records.json') != done['records_sha256']:
        raise ValueError('Invalid indexed reference')
    source = Path(read(a.merged / 'locations.json')[a.sequence])
    old = read(source / 'records.json')[0]
    prompt = source / a.sequence / 'prompt.json'
    if digest(prompt) != old['prompt_sha256']:
        raise ValueError('Prompt changed')
    first = read(prompt)['first_detection_index']
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'), **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    frames = videos[a.sequence]
    target = sorted(set([first + 1] + [first + round((len(frames) - 1 - first) * i / 4) for i in range(1, 5)]))
    if not target or target[0] <= first or target[-1] >= len(frames):
        raise ValueError('Invalid frame schedule')
    events = []
    original = runner.indexed_spatial_rope
    @contextmanager
    def capture(model, positions):
        with original(model, positions):
            with capture_memory_read(model, set(target), a.output, events):
                yield
    runner.indexed_spatial_rope = capture
    sys.argv = ['run_lean_indexed_spatial_rope.py', '--mode', 'gate' if a.label_free_gate else 'candidate', '--sequence', a.sequence]
    for key in ('checkpoint', 'merged', 'reference', 'cpu_verification', 'observed_reference', 'output'):
        sys.argv.extend(['--' + key.replace('_', '-'), str(getattr(a, key))])
    sys.argv.extend(['--gate-a', str(a.indexed_gate_a), '--gate-b', str(a.indexed_gate_b)])
    runner.main()
    record = read(a.output / 'records.json')
    keys = ('mask_sha256', 'positions_sha256', 'selections_sha256', 'memory_events_sha256')
    if not a.label_free_gate:
        keys += ('frames_sha256',)
    for key in keys:
        if record[key] != ref[key]:
            raise ValueError('Capture changed reference: ' + key)
    if [x['frame'] for x in events] != target:
        raise ValueError('Capture coverage differs')
    for event in events:
        event['sha256'] = digest(a.output / 'snapshots' / event['snapshot_file'])
    result = {'passed': True, 'sequence': a.sequence, 'target_frames': target, 'snapshots': events,
              'reference_records_sha256': done['records_sha256'], 'records_sha256': digest(a.output / 'records.json'),
              'protocol_sha256': digest(PROTOCOL), 'source_sha256': digest(Path(__file__)),
              'capture_sha256': digest(ROOT / 'modeling/lean_memory_capture.py')}
    (a.output / 'capture_observation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
