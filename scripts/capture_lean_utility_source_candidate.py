#!/usr/bin/env python3
"""Capture verified source provenance on an existing sequence-separated trajectory."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import runpy
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import modeling.lean_memory_capture as memory_capture
from modeling.lean_utility_source_capture import capture_utility_sources
from scripts.memory_token_io import digest
p = argparse.ArgumentParser(add_help=False)
p.add_argument('--source-output', type=Path, required=True)
p.add_argument('--source-verification', type=Path, required=True)
a, rest = p.parse_known_args()
if '--output' in rest or '--mode' not in rest or rest[rest.index('--mode')+1] != 'candidate':
    raise ValueError('Candidate mode required')
v = json.loads((a.source_verification/'verification.json').read_text())
if not v['passed'] or v['observer_sha256'] != digest(ROOT/'modeling/lean_utility_source_capture.py') or v['runner_sha256'] != digest(ROOT/'scripts/capture_lean_utility_source_gate.py') or v['verifier_sha256'] != digest(ROOT/'scripts/verify_lean_utility_source_gates.py'):
    raise ValueError('Verified source engineering gate required')
original = memory_capture.capture_memory_read; events = []
@contextmanager
def nested(model, targets, output, observations):
    with original(model, targets, output, observations), capture_utility_sources(model, targets, output, events):
        yield
memory_capture.capture_memory_read = nested
old_argv = sys.argv
try:
    sys.argv = [str(ROOT/'scripts/capture_lean_utility_training_states.py'), *rest, '--output', str(a.source_output)]
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    memory_capture.capture_memory_read = original; sys.argv = old_argv
r = json.loads((a.source_output/'utility_capture_receipt.json').read_text())
if not r['passed'] or {e['frame'] for e in events} != set(r['targets']):
    raise ValueError('Source coverage differs')
for frame in r['targets']:
    target = torch.load(a.source_output/'features'/f'{frame:06d}.pt', map_location='cpu', weights_only=True)
    source = torch.load(a.source_output/'source_features'/f'{frame:06d}.pt', map_location='cpu', weights_only=True)
    if not torch.equal(target['feature'], source['feature']):
        raise ValueError('Source and target early features differ')
result = {'passed': True, 'sequence': r['sequence'], 'observations': events,
          'source_features': {p.name: digest(p) for p in sorted((a.source_output/'source_features').glob('*.pt'))},
          'source_snapshots': {p.name: digest(p) for p in sorted((a.source_output/'source_snapshots').glob('*.pt'))},
          'nested_receipt_sha256': digest(a.source_output/'utility_capture_receipt.json'),
          'source_gate_sha256': digest(a.source_verification/'verification.json'),
          'observer_sha256': digest(ROOT/'modeling/lean_utility_source_capture.py'),
          'runner_sha256': digest(Path(__file__))}
(a.source_output/'source_capture_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'passed': True, 'sequence': r['sequence'], 'observations': events}))
