#!/usr/bin/env python3
"""Explore multiscale projected memory similarities against saved tile utility."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.multiscale_memory_utility import multiscale_scores, attention_tile_mass, evaluate_scores
from modeling.signed_memory_feature_attribution import SpatialMemoryScalar
from scripts.memory_token_io import digest, parameter_digest
from scripts.run_signed_memory_feature_utility_overlap import activate_model, verify_pins, tensor_sha

PROTOCOL = ROOT / 'protocols/multiscale_memory_utility_diagnostic_v1.json'


def checked_inputs(protocol):
    root = Path(protocol['capture_archive'])
    for name, sha in protocol['archived_receipt_hashes'].items():
        if digest(root / name) != sha:
            raise RuntimeError(f'Prior receipt changed: {name}')
    if not json.loads((root / 'audit.json').read_text())['passed']:
        raise RuntimeError('Prior independent audit was not passed')
    signed = json.loads((ROOT / protocol['pairs_protocol']).read_text())
    _, utility_root = verify_pins(signed)
    receipts = json.loads((root / 'capture_receipts.json').read_text())
    aligned = json.loads((root / 'utility_alignment.json').read_text())['pairs']
    if len(receipts) != len(signed['pairs']) or len(aligned) != len(receipts):
        raise RuntimeError('Prior capture coverage differs')
    cases = []
    for pair, receipt, row in zip(signed['pairs'], receipts, aligned, strict=True):
        key = (pair['sequence'], pair['source_frame'], pair['target_frame'])
        if key != (receipt['sequence'], receipt['source_frame'], receipt['target_frame']) or key != (row['sequence'], row['source_frame'], row['target_frame']):
            raise RuntimeError('Prior source/target mismatch')
        if digest(Path(receipt['capture_path'])) != receipt['capture_sha256']:
            raise RuntimeError('Captured native read changed')
        archived = json.loads((utility_root / 'scientific_run_20261108' / 'sequences' / pair['sequence'] / 'results.json').read_text())
        record = next(x for x in archived['main'] if x.get('anchor_target') == pair['target_frame'] and x.get('source_frame') == pair['source_frame'])
        arms = record['arms']
        if arms['native']['prediction_sha256'] != pair['expected_native_prediction_sha256']:
            raise RuntimeError('Archived native prediction mismatch')
        utility = []
        for tile in range(16):
            arm = arms[f'tile_{tile:02d}']
            if [x['source_frame'] for x in arm['read']['selection_layers']] != [pair['source_frame']] * 4:
                raise RuntimeError('Tile intervention source or layer mismatch')
            utility.append(float(arms['native']['scores']['dice']) - float(arm['scores']['dice']))
        if not np.allclose(utility, row['tile_removal_utility'], rtol=0, atol=1e-12):
            raise RuntimeError('Tile utility differs from previously audited values')
        cases.append((pair, receipt, utility))
    return signed, cases


def inspect_read(model, capture):
    """Capture native projected first-layer Q/K and post-RoPE Q/K, retaining native output."""
    scalar = SpatialMemoryScalar(model, SimpleNamespace(memory=capture['memory'], decoder=capture['decoder'], outputs=capture['outputs'], layers=capture['layers']))
    first = model.memory_attention.layers[0].cross_attn_image
    projected, rope, seen = {}, {}, []
    active = [False]
    def pre(module, args, kwargs):
        if active[0]:
            raise RuntimeError('Nested first-layer attention')
        active[0] = True
    def post(module, args, kwargs, output):
        active[0] = False
    def save(name):
        def hook(module, args, output):
            projected[name] = output.detach().cpu().clone()
        return hook
    def observe(q, k, v, *args, **kwargs):
        if active[0]:
            if rope:
                raise RuntimeError('Multiple first-layer attention calls')
            rope['q'], rope['k'] = q.detach().cpu().clone(), k.detach().cpu().clone()
        return native_sdpa(q, k, v, *args, **kwargs)
    handles = [first.register_forward_pre_hook(pre, with_kwargs=True), first.register_forward_hook(post, with_kwargs=True),
               first.q_proj.register_forward_hook(save('q')), first.k_proj.register_forward_hook(save('k'))]
    for idx, layer in enumerate(model.memory_attention.layers):
        handles.append(layer.register_forward_hook(lambda module, args, out, idx=idx: seen.append(idx)))
    native_sdpa = F.scaled_dot_product_attention
    memory_before = tensor_sha(scalar.memory['memory'])
    try:
        with torch.no_grad(), patch.object(F, 'scaled_dot_product_attention', observe):
            outputs = scalar.raw_outputs(scalar.inputs)
    finally:
        for handle in handles:
            handle.remove()
    if seen != [0, 1, 2, 3] or active[0] or set(projected) != {'q', 'k'} or set(rope) != {'q', 'k'}:
        raise RuntimeError('Native read did not capture the expected four-layer route')
    if any(not torch.equal(a, b) for a, b in zip(outputs, capture['outputs'], strict=True)):
        raise RuntimeError('Native decoder replay is not exact')
    if tensor_sha(scalar.memory['memory']) != memory_before:
        raise RuntimeError('Capture memory changed')
    return projected, rope


def run(output):
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(PROTOCOL.read_text())
    signed, cases = checked_inputs(protocol)
    model, revision = activate_model(signed)
    if digest(Path(signed['checkpoint'])) != protocol['checkpoint_sha256']:
        raise RuntimeError('Checkpoint mismatch')
    before = parameter_digest(model)
    output.mkdir(parents=True)
    setup = {'protocol_sha256': digest(PROTOCOL), 'code_sha256': digest(Path(__file__)),
             'descriptor_sha256': digest(ROOT / 'modeling/multiscale_memory_utility.py'),
             'checkpoint_sha256': protocol['checkpoint_sha256'], 'sam2_revision': revision,
             'pairs': len(cases), 'mode': 'read-only captured native replay; no intervention', 'started_unix': time.time()}
    (output / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    results = []
    for pair, receipt, utility in cases:
        start_time = time.monotonic()
        capture = torch.load(receipt['capture_path'], map_location='cpu', weights_only=True)
        if capture['metadata']['native_prediction_sha256'] != pair['expected_native_prediction_sha256']:
            raise RuntimeError('Native capture prediction mismatch')
        projected, rope = inspect_read(model, capture)
        spans = [s for s in capture['metadata']['memory_sources'] if s['frame'] == pair['source_frame']]
        if len(spans) != 1:
            raise RuntimeError('Target source span is not unique')
        source = spans[0]
        begin, end = source['start'], source['stop']
        q, k = projected['q'][0], projected['k'][0, begin:end]
        if q.shape[0] != 4096 or k.shape != q.shape:
            raise RuntimeError(f'Projected query/source shapes mismatch: {q.shape} {k.shape}')
        scores = multiscale_scores(q, k)
        scores['native_attention'] = attention_tile_mass(rope['q'], rope['k'], begin, end)
        if before != parameter_digest(model):
            raise RuntimeError('SAM2 parameters changed')
        metrics = {name: evaluate_scores(scores[name], utility) for name in ('scale_1', 'multiscale', 'native_attention')}
        results.append({'sequence': pair['sequence'], 'source_frame': pair['source_frame'], 'target_frame': pair['target_frame'],
                        'native_prediction_sha256': pair['expected_native_prediction_sha256'], 'source_span': [begin, end],
                        'scores': {name: values.tolist() for name, values in scores.items()}, 'tile_removal_utility': utility,
                        'metrics': metrics, 'elapsed_seconds': time.monotonic() - start_time})
        (output / 'progress.json').write_text(json.dumps({'completed': len(results), 'expected': len(cases), 'last_pair': pair['sequence']}) + '\n')
    summary = {}
    for name in ('scale_1', 'multiscale', 'native_attention'):
        values = [r['metrics'][name] for r in results]
        rho = [r['spearman_rho'] for r in values if r['spearman_rho'] is not None]
        summary[name] = {'mean_spearman': float(np.mean(rho)) if rho else None,
                         'defined_pair_count': len(rho),
                         'equal_pair_selected_minus_uniform_dice': float(np.mean([r['selected_minus_uniform_dice'] for r in values]))}
    report = {'scope': protocol['scope'], 'pairs': results, 'equal_pair_summary': summary,
              'limitations': protocol['limitations'], 'efficacy_evaluated': False}
    result = output / 'results.json'
    result.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'completion.json').write_text(json.dumps({'complete': True, 'results_sha256': digest(result), 'setup_sha256': digest(output / 'setup.json')}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == '__main__':
    main()
