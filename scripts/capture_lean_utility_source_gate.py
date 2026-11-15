#!/usr/bin/env python3
"""Engineering-only nested observer gate; does not run a token intervention."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import runpy
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import modeling.lean_memory_capture as memory_capture
from modeling.lean_utility_source_capture import capture_utility_sources
from scripts.memory_token_io import digest

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--source-output', type=Path, required=True)
args, rest = parser.parse_known_args()
if '--output' in rest or '--mode' not in rest or rest[rest.index('--mode') + 1] != 'gate':
    raise ValueError('This wrapper only permits the independent seq3 engineering gate')
original = memory_capture.capture_memory_read
observations = []
@contextmanager
def nested(model, targets, output, events):
    with original(model, targets, output, events), capture_utility_sources(model, targets, output, observations):
        yield
memory_capture.capture_memory_read = nested
old_argv = sys.argv
try:
    sys.argv = [str(ROOT / 'scripts/capture_lean_utility_training_states.py'), *rest, '--output', str(args.source_output)]
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    memory_capture.capture_memory_read = original
    sys.argv = old_argv
receipt = json.loads((args.source_output / 'utility_capture_receipt.json').read_text())
if not receipt['passed'] or {e['frame'] for e in observations} != set(receipt['targets']):
    raise ValueError('Source coverage differs')
for frame in receipt['targets']:
    target = torch.load(args.source_output / 'features' / f'{frame:06d}.pt', map_location='cpu', weights_only=True)
    source = torch.load(args.source_output / 'source_features' / f'{frame:06d}.pt', map_location='cpu', weights_only=True)
    if not torch.equal(target['feature'], source['feature']):
        raise ValueError('Source and target early feature differ')
source_files = sorted((args.source_output / 'source_features').glob('*.pt'))
source_snapshots = sorted((args.source_output / 'source_snapshots').glob('*.pt'))
result = {'passed': True, 'observations': observations,
          'source_features': {p.name: digest(p) for p in source_files},
          'source_snapshots': {p.name: digest(p) for p in source_snapshots},
          'nested_receipt_sha256': digest(args.source_output / 'utility_capture_receipt.json'),
          'observer_sha256': digest(ROOT / 'modeling/lean_utility_source_capture.py'),
          'runner_sha256': digest(Path(__file__))}
(args.source_output / 'source_capture_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'source_capture_passed': True, 'observations': observations}))
