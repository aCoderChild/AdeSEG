#!/usr/bin/env python3
"""Run a fixed, CMSA-inspired context variant using the released Lean runner."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest  # noqa: E402
from modeling.lean_multiscale_anchor import install_multiscale_anchor  # noqa: E402
from scripts.memory_token_io import digest, prompt_index, validate_output  # noqa: E402
from scripts.run_author_baseline import (  # noqa: E402
    MODEL_CONFIG, assert_parameter_integrity, build_author_predictor,
    parameter_fingerprints, rollout, source_identity,
)

PROTOCOL = ROOT / 'protocols/lean_multiscale_anchor_v1.json'
MANIFEST = ROOT / 'data/polypgen_sequence.jsonl'
IMPLEMENTATION = ROOT / 'modeling/lean_multiscale_anchor.py'


class NoNetworkLogger:
    def log(self, *args, **kwargs):
        pass


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True, help='Existing full Lean author output')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sequence', action='append', required=True)
    p.add_argument('--mode', choices=('gate', 'treatment'), required=True)
    p.add_argument('--gate-a', type=Path)
    p.add_argument('--gate-b', type=Path)
    args = p.parse_args()
    if len(set(args.sequence)) != len(args.sequence):
        raise ValueError('Duplicate sequence')
    repo, revision = source_identity('lean')
    baseline_setup = json.loads((args.reference / 'setup.json').read_text())
    if not json.loads((args.reference / 'delivery.json').read_text())['completed']:
        raise ValueError('Lean reference incomplete')
    required = {'method': 'lean', 'author_revision': revision, 'checkpoint_sha256': digest(args.checkpoint),
                'manifest_sha256': digest(MANIFEST), 'runner_sha256': digest(ROOT / 'scripts/run_author_baseline.py'),
                'device': 'cpu', 'torch': torch.__version__, 'config': MODEL_CONFIG, 'memory_stride': 1,
                'apply_postprocessing': False}
    if any(baseline_setup.get(k) != v for k, v in required.items()):
        raise ValueError('Lean reference is not matched to current runtime, source and checkpoint')
    signature = {'lean': required, 'protocol_sha256': digest(PROTOCOL), 'implementation_sha256': digest(IMPLEMENTATION),
                 'runner_sha256': digest(Path(__file__)), 'baseline_setup_sha256': digest(args.reference / 'setup.json'),
                 'bypass_sha256': digest(repo / 'checkpoints/bypass_bottleneck_base.pth'),
                 'lean_config_sha256': digest(repo / 'configs/lean_sam2/lean_sam2.yaml')}
    if args.mode == 'treatment':
        if not args.gate_a or not args.gate_b:
            raise ValueError('Two independently executed gates are required')
        gates = []
        for gate in (args.gate_a, args.gate_b):
            meta = json.loads((gate / 'setup.json').read_text())
            receipt = json.loads((gate / 'completion.json').read_text())
            if meta['signature'] != signature or meta['mode'] != 'gate' or not receipt['passed'] or digest(gate / 'records.json') != receipt['records_sha256']:
                raise ValueError('Gate identity or receipt mismatch')
            gates.append(json.loads((gate / 'records.json').read_text()))
        if gates[0] != gates[1] or not gates[0]:
            raise ValueError('Independent gate records differ')
    videos = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    output = validate_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'setup.json').write_text(json.dumps({'signature': signature, 'mode': args.mode, 'sequences': args.sequence}, indent=2) + '\n')
    protocol = json.loads((ROOT / 'protocols/memory_token_regions.json').read_text())
    torch.set_num_threads(2)
    records = []
    for name in args.sequence:
        frames = videos[name]
        prompt = prompt_index(frames, protocol)
        baseline_inputs = json.loads((args.reference / name / 'inputs.json').read_text())
        for frame, ref in zip(frames, baseline_inputs):
            if frame.frame_index != ref['frame'] or digest(frame.image_path) != ref['image_sha256'] or digest(frame.mask_path) != ref['mask_sha256']:
                raise ValueError(f'Input mismatch: {name}/{frame.frame_index}')
        if len(frames) != len(baseline_inputs):
            raise ValueError(f'Input length mismatch: {name}')
        baseline = read_lines(args.reference / name / 'frames.jsonl')
        for arm, strength in ([('zero', 0.), ('multiscale', .5)] if args.mode == 'gate' else [('multiscale', .5)]):
            folder = output / name / arm
            folder.mkdir(parents=True)
            predictor = build_author_predictor(SimpleNamespace(method='lean', checkpoint=args.checkpoint, device='cpu'), repo)
            before = parameter_fingerprints(predictor)
            selections = []
            events = install_multiscale_anchor(predictor, strength, selection_events=selections)
            rows, _ = rollout(predictor, 'lean', frames, prompt, folder, 'cpu', NoNetworkLogger())
            assert_parameter_integrity(predictor, before, folder, 'after')
            (folder / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
            (folder / 'selections.json').write_text(json.dumps(selections, indent=2) + '\n')
            if len(rows) != len(baseline) or any((x['frame'], x['source_frame_index'], x['present']) != (y['frame'], y['source_frame_index'], y['present']) for x, y in zip(rows, baseline)):
                raise ValueError('Target frames or labels differ from Lean reference')
            if arm == 'zero' and any(x['prediction_sha256'] != y['prediction_sha256'] for x, y in zip(rows, baseline)):
                raise ValueError('Zero-strength hook does not reproduce original Lean logits')
            if arm == 'multiscale' and not events:
                raise ValueError('Multiscale anchor hook was never activated')
            if not selections:
                raise ValueError('Lean did not perform any token selection')
            records.append({'sequence': name, 'arm': arm, 'target_count': len(rows),
                            'frames_sha256': digest(folder / 'frames.jsonl'), 'events_sha256': digest(folder / 'events.json'),
                            'selections_sha256': digest(folder / 'selections.json'),
                            'source_logit_hashes': [r['prediction_sha256'] for r in rows],
                            'events_count': len(events), 'mean_score_change': float(np.mean([e['mean_abs_score_change'] for e in events])) if events else 0.})
            del predictor
    (output / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
    (output / 'completion.json').write_text(json.dumps({'passed': True, 'records_sha256': digest(output / 'records.json')}, indent=2) + '\n')
    print(f'Completed {args.mode}: {args.sequence}', flush=True)


if __name__ == '__main__':
    main()
