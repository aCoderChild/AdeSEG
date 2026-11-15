"""Reuse the real DAVIS persistent-ID gate for the ToMe/ATC read integration.

Checks registration-order and isolated object-bank invariance. No accuracy or
cross-condition efficacy claim: each condition has its own propagated history.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.vos_benchmarks import inventory
from modeling.author_token_matching import use_matching
from modeling.pitome_matching import use_pitome, author_pitome, PITOME_SOURCE
from modeling.region_constrained_tome import NativeRegionConstrainedToMe
from scripts import verify_position_consistent_tome_davis_ids as legacy
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_regional_refinement import write


class RoutingRecorder(NativeRegionConstrainedToMe):
    @property
    def provenance_reads(self):
        # Legacy rollout serializes this field. Report actual routing/budget
        # records, explicitly without pretending to have cluster-member traces.
        return [{'scope': 'routing and token budgets only; no cluster-membership trace',
                 'read': call} for call in self.calls if call['frame'] == 1]


def check_isolation(results, prefix):
    joint = results[prefix + '_joint']
    reverse = results[prefix + '_reverse']
    fields = ('stored_mask_sha256', 'stored_memory_sha256', 'object_score_sha256')
    for frame in range(legacy.FRAME_LIMIT):
        for object_id in legacy.OBJECT_IDS:
            key = str(object_id)
            a = joint['frames'][str(frame)][key]
            b = reverse['frames'][str(frame)][key]
            isolated = results[prefix + f'_id{object_id}']['frames'][str(frame)][key]
            if any(a[field] != b[field] or a[field] != isolated[field] for field in fields):
                raise RuntimeError(f'Object bank isolation/order failed: {prefix}/{frame}/{object_id}')
            # Prompt-frame joint consolidation can apply non-overlap rules;
            # only propagated logits support an output-isolation claim here.
            if frame > 0 and (a['logits_sha256'] != b['logits_sha256'] or
                              a['logits_sha256'] != isolated['logits_sha256']):
                raise RuntimeError(f'Propagated output isolation failed: {prefix}/{frame}/{object_id}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'protocols/author_matching_ids_gate_v1.json')
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        protocol_path = args.protocol
        protocol = json.loads(protocol_path.read_text())
        if any(condition.startswith('pitome_') for condition in protocol['conditions']):
            pitome_repo = Path(protocol['pitome_repository'])
            author_pitome(pitome_repo)
            if digest(pitome_repo / PITOME_SOURCE) != protocol['pitome_source_sha256']:
                raise RuntimeError('PiToMe source differs from the engineering pin')
            if protocol['merge_divisor'] != 4:
                raise RuntimeError('Protection gate requires the frozen quarter-group budget')
        torch.set_num_threads(2)
        torch.manual_seed(20261111)
        torch.use_deterministic_algorithms(True)
        checkpoint = Path(protocol['checkpoint'])
        if digest(checkpoint) != protocol['checkpoint_sha256']:
            raise RuntimeError('Checkpoint mismatch')
        repo = ROOT / protocol['sam2_repository']
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != protocol['sam2_revision']:
            raise RuntimeError('SAM2 revision mismatch')
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip():
            raise RuntimeError('SAM2 checkout differs from the pinned clean source')
        if 'sam2' in sys.modules:
            raise RuntimeError('SAM2 imported before source activation')
        sys.path.insert(0, str(repo))
        inv = inventory(legacy.DAVIS_ROOT, 'davis2017', 'val')
        record = next(v for v in inv['videos'] if v['video'] == legacy.VIDEO)
        if record['object_ids'] != list(legacy.OBJECT_IDS):
            raise RuntimeError('Fixture object IDs changed')
        hashes = {p: digest(ROOT / p) for p in protocol['source_files']}
        write(out / 'setup.json', {'protocol': protocol, 'protocol_sha256': digest(protocol_path),
              'sources': hashes, 'davis_val_list_sha256': digest(legacy.DAVIS_ROOT / 'ImageSets/2017/val.txt')})
        model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu',
            checkpoint=checkpoint, model_config=protocol['model_config']), repo)
        initial_parameters = parameter_digest(model)
        results = {}
        for condition in ['native'] + protocol['conditions']:
            for label, order in [('joint', (1, 2)), ('reverse', (2, 1)), ('id1', (1,)), ('id2', (2,))]:
                key = condition + '_' + label
                write(out / 'progress.json', {'condition': condition, 'rollout': label,
                      'completed': len(results), 'total': 20, 'elapsed_seconds': time.monotonic()-started})
                if condition == 'native':
                    result = legacy.rollout(model, record, out / key, order, use_tome=False)
                else:
                    strategy, mode = condition.split('_')
                    def factory(*args, **kwargs):
                        kwargs.pop('provenance_target', None)
                        kwargs.pop('hash_tensors_target', None)
                        kwargs['mode'] = mode
                        kwargs['merge_divisor'] = protocol.get('merge_divisor', 2)
                        return RoutingRecorder(*args, **kwargs)
                    context = use_pitome(pitome_repo) if strategy == 'pitome' else use_matching(strategy)
                    with patch.object(legacy, 'PositionConsistentRegionToMe', factory), context:
                        result = legacy.rollout(model, record, out / key, order, use_tome=True)
                results[key] = result
                write(out / f'{key}.json', result)
            check_isolation(results, condition)
            write(out / f'{condition}_isolation.json', {'passed': True})
        if parameter_digest(model) != initial_parameters:
            raise RuntimeError('Frozen SAM2 parameters changed')
        if any(digest(ROOT / p) != sha for p, sha in hashes.items()):
            raise RuntimeError('Gate source changed during execution')
        write(out / 'completion.json', {'passed': True, 'conditions': ['native']+protocol['conditions'],
              'scope': 'DAVIS bike-packing IDs 1,2, frames 0-2; routing engineering only',
              'registration_order_invariant': True, 'isolated_stored_banks_identical': True,
              'propagated_logits_isolated': True, 'all_four_layers_per_object_target': True,
              'parameters_unchanged': True, 'source_hashes_unchanged': True,
              'cross_condition_budgets_or_accuracy_compared': False,
              'elapsed_seconds': time.monotonic()-started})
        write(out / 'progress.json', {'phase': 'complete', 'completed': 20, 'total': 20})
    except Exception as exc:
        write(out / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()
