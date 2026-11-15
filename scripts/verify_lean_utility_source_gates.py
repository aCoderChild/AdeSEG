#!/usr/bin/env python3
"""Independently verify reproducible source identities on frozen attention reads."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from modeling.lean_memory_position_lookup import MemoryPositionLookup
from modeling.lean_region_read_intervention import locate_verified_span
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, parameter_fingerprints, assert_parameter_integrity
from scripts.run_lean_token_window import FORK
from scripts.run_lean_offline_memory_ablation import equal

p = argparse.ArgumentParser()
for name in ('first', 'second', 'checkpoint', 'nested-verification', 'model-verification', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
read = lambda f: json.loads(f.read_text())
x = read(a.first/'source_capture_receipt.json'); y = read(a.second/'source_capture_receipt.json')
v = read(a.nested_verification/'verification.json')
m = read(a.model_verification/'verification.json')
if x != y or not x['passed'] or not v['passed'] or v['verifier_sha256'] != digest(ROOT/'scripts/verify_lean_utility_training_gates.py') or v['first_receipt_sha256'] != digest(a.first/'utility_capture_receipt.json') or v['second_receipt_sha256'] != digest(a.second/'utility_capture_receipt.json') or not m['passed'] or m['verifier_sha256'] != digest(ROOT/'scripts/verify_lean_token_window_gates.py') or m['identity']['checkpoint_sha256'] != digest(a.checkpoint) or x['runner_sha256'] != digest(ROOT/'scripts/capture_lean_utility_source_gate.py') or x['observer_sha256'] != digest(ROOT/'modeling/lean_utility_source_capture.py'):
    raise ValueError('Source gate receipts differ or code changed')
for folder in (a.first, a.second):
    if digest(folder/'utility_capture_receipt.json') != x['nested_receipt_sha256']:
        raise ValueError('Nested receipt changed')
    for group in ('source_features', 'source_snapshots'):
        files = x[group]
        if not files or {p.name for p in (folder/group).glob('*.pt')} != set(files):
            raise ValueError('Source file coverage changed')
        for name, expected in files.items():
            if digest(folder/group/name) != expected:
                raise ValueError('Source file changed')
out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2); sys.path.insert(0, str(FORK))
model = build_author_predictor(SimpleNamespace(method='lean', checkpoint=a.checkpoint, device='cpu'), FORK)
before = parameter_fingerprints(model)
with torch.no_grad():
    grid = model.memory_encoder.position_encoding(torch.zeros(1, model.mem_dim, 64, 64)).flatten(2).permute(2, 0, 1)[:, 0].cpu().float()
lookup = MemoryPositionLookup(grid, torch.zeros(1, model.mem_dim))
offsets = model.maskmem_tpos_enc.detach().reshape(-1, model.mem_dim).cpu()
for item in x['observations']:
    frame = item['frame']
    data = torch.load(a.first/'snapshots'/f'{frame:06d}.pt', map_location='cpu', weights_only=True)
    source = torch.load(a.first/'source_snapshots'/f'{frame:06d}.pt', map_location='cpu', weights_only=True)
    second = torch.load(a.second/'source_snapshots'/f'{frame:06d}.pt', map_location='cpu', weights_only=True)
    if not equal(source, second) or frame != source['frame'] or source['pointer_count'] != data['attention_inputs']['num_obj_ptr_tokens']:
        raise ValueError('Source state differs')
    args = data['attention_inputs']; protected = [(0, 4096)]; count = 0
    for s in source['sources']:
        if s['frame'] >= frame or not (a.first/'source_features'/f"{s['frame']:06d}.pt").is_file():
            raise ValueError('Source frame invalid or early features missing')
        start = locate_verified_span(args['memory'], args['memory_pos'], source['pointer_count'], s['features'], s['positions'], offsets, protected)
        ids = lookup.decode(s['positions'].float(), 0)
        if start != s['start'] or not torch.equal(ids, s['token_ids']) or len(torch.unique(ids)) != len(ids):
            raise ValueError('Source span or spatial identities changed')
        protected.append((start, start+len(ids))); count += len(ids)
    if len(source['sources']) != item['attributed_sources'] or count != item['attributed_tokens'] or source['rejected'] != item['rejected']:
        raise ValueError('Attribution counts differ')
assert_parameter_integrity(model, before, out, 'after')
result = {'passed': True, 'first_sha256': digest(a.first/'source_capture_receipt.json'),
          'second_sha256': digest(a.second/'source_capture_receipt.json'), 'checkpoint_sha256': digest(a.checkpoint),
          'observer_sha256': x['observer_sha256'], 'runner_sha256': x['runner_sha256'],
          'verifier_sha256': digest(Path(__file__)), 'observations': x['observations'],
          'meaning': 'Engineering provenance only; no labels, utility or inference gain established.'}
(out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
print('Independent source identity, span and spatial-index verification passed.')
