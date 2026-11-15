"""Post-completion audit of where native mask roles can constrain local merging.

Counts constrainable windows, not helpful tokens or causal contributions.
Reuses the exact role-assignment helper and verified archived native banks.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.native_history_cache import load_bank
from modeling.soft_role_affinity import mask_logits_to_soft_roles
from scripts.memory_token_io import digest, validate_output


def window_scope(roles, window_size=4, merge_divisor=2):
    if type(merge_divisor) is not int or merge_divisor < 2:
        raise ValueError('Merge divisor must be an integer of at least two')
    if roles.ndim != 3 or roles.shape[-1] != 3 or roles.shape[0] != roles.shape[1]:
        raise ValueError('Expected square spatial grid with three role channels')
    side = roles.shape[0]
    if window_size <= 0 or side % window_size:
        raise ValueError('Windows must tile the spatial grid')
    if not torch.isfinite(roles).all() or (roles < 0).any() or not torch.allclose(roles.sum(-1), torch.ones_like(roles[..., 0]), atol=1e-6):
        raise ValueError('Invalid role memberships')
    labels = roles.argmax(-1)
    windows = labels.reshape(side//window_size, window_size, side//window_size, window_size)
    windows = windows.permute(0, 2, 1, 3).reshape(-1, window_size**2)
    counts = F.one_hot(windows, 3).sum(1)
    distinct = (counts > 0).sum(1)
    merges = (counts//merge_divisor).sum(1)
    mixed = distinct > 1
    return {'windows': len(windows), 'mixed_role_windows': int(mixed.sum()),
            'mixed_windows_with_unchanged_budget': int((mixed & (merges == window_size**2//merge_divisor)).sum()),
            'role_token_counts': F.one_hot(labels, 3).sum((0, 1)).tolist(),
            'merges_per_window': merges.tolist()}


def report(folder, plan_path):
    folder = validate_output(folder)
    complete = json.loads((folder / 'completion.json').read_text())
    queued = json.loads((folder.parent / 'completion.json').read_text())
    if not complete.get('passed') or complete['scope'] != 'broader_causal_diagnostic' or not queued.get('complete'):
        raise ValueError('Completed primary experiment and analysis required')
    if digest(plan_path) != complete['plan_sha256'] or digest(folder / 'results.json') != complete['results_sha256']:
        raise ValueError('Completed plan/results changed')
    primary_sha = digest(folder / 'summary.json')
    if primary_sha != queued['summary_sha256']:
        raise ValueError('Frozen analysis summary changed')
    plan = json.loads(plan_path.read_text())
    protocol = plan['protocol']
    merge_divisor = protocol.get('merge_divisor', 2)
    for source in ('modeling/soft_role_affinity.py', 'modeling/prediction_conditioned_affinity.py',
                   'modeling/native_history_cache.py'):
        if digest(ROOT / source) != plan['source_hashes'][source]:
            raise ValueError(f'Role/cache helper changed after the experiment: {source}')
    records = json.loads((folder / 'results.json').read_text())
    result = {}
    for sequence in protocol['sequences']:
        specification = plan['sequences'][sequence]
        cache = Path(specification['cache_directory'])
        if digest(cache / 'receipt.json') != specification['receipt_sha256']:
            raise ValueError('Source-bank receipt changed')
        receipt = json.loads((cache / 'receipt.json').read_text())
        payload = load_bank(cache / 'bank.pt', receipt, checkpoint_sha256=protocol['checkpoint_sha256'],
            manifest_sha256=protocol['manifest_sha256'], model_config=protocol['model_config'])
        multiplicity = Counter()
        source_stats = {}
        for row in records[sequence]:
            reads = row['attention_audit']['tome_regional']
            if [read['layer'] for read in reads] != [0, 1, 2, 3]:
                raise ValueError('Missing native matching layer evidence')
            for frame in reads[0]['frames']:
                index = frame['frame']
                if index >= row['frame']:
                    raise ValueError('Source-frame causality violated')
                multiplicity[index] += 1
                if index not in source_stats:
                    stored = payload['outputs']['cond_frame_outputs'].get(index,
                             payload['outputs']['non_cond_frame_outputs'].get(index))
                    if stored is None:
                        raise ValueError('Logged memory frame missing from archived bank')
                    scope = window_scope(mask_logits_to_soft_roles(stored['pred_masks']),
                                         merge_divisor=merge_divisor)
                    scope['predicted_empty'] = not bool(stored['pred_masks'].gt(0).any())
                    source_stats[index] = scope
                computed = source_stats[index]
                if computed['role_token_counts'] != frame['region_token_counts']:
                    raise ValueError('Reconstructed roles differ from the executed read')
                if computed['merges_per_window'] != [w['merges'] for w in frame['windows']]:
                    raise ValueError('Reconstructed local token budget differs')
        total_windows = sum(source_stats[index]['windows'] * count for index, count in multiplicity.items())
        mixed_windows = sum(source_stats[index]['mixed_role_windows'] * count for index, count in multiplicity.items())
        empty_reads = sum(count for index, count in multiplicity.items() if source_stats[index]['predicted_empty'])
        result[sequence] = {'unique_sources': len(source_stats), 'source_frame_reads': sum(multiplicity.values()),
            'predicted_empty_source_reads': empty_reads, 'window_reads': total_windows,
            'mixed_role_window_reads': mixed_windows,
            'mixed_window_fraction': mixed_windows/total_windows,
            'mixed_same_budget_window_reads': sum(source_stats[index]['mixed_windows_with_unchanged_budget']*count for index, count in multiplicity.items()),
            'sources': {str(index): {**value, 'source_read_count': multiplicity[index]}
                        for index, value in source_stats.items()}}
        del payload
    payload = {'scope': 'Descriptive mask-role constraint opportunity, not token usefulness',
               'merge_divisor': merge_divisor,
               'role_order': ['background', 'foreground', 'within_cell_mixedness'],
               'source_read_weighting': 'One count per target/source; identical layer repetitions omitted',
               'reporter_sha256': digest(Path(__file__)), 'primary_summary_sha256': primary_sha,
               'role_helper_sha256': digest(ROOT / 'modeling/soft_role_affinity.py'),
               'sequences': result}
    (folder / 'region_constraint_scope.json').write_text(json.dumps(payload, indent=2)+'\n')
    lines = ['# Scope of the regional merge constraint', '',
             'Reconstructed from the exact native stored masks and role helper, with role counts and merge budgets checked against the executed first-layer read. Homogeneous-role windows impose no region-based restriction. Mixed windows are opportunities for the constraint to matter; they do not prove that generic matching actually crossed a region.', '',
             '| Sequence | Unique sources | Source-frame reads | Predicted-empty source reads | Mixed-role window reads / all | Fraction |',
             '|---|---:|---:|---:|---:|---:|']
    for sequence, value in result.items():
        lines.append(f"| {sequence} | {value['unique_sources']} | {value['source_frame_reads']} | {value['predicted_empty_source_reads']} | {value['mixed_role_window_reads']}/{value['window_reads']} | {value['mixed_window_fraction']:.2%} |")
    lines += ['', 'Role channels describe predicted-mask footprints, not exclusive feature semantics. A predicted-empty source is defined by the native mask decision, not a new confidence threshold. Read weighting includes repeated use of the same source frame. The audit introduces no inference changes or additional candidate-selection rules.', '']
    (folder / 'region_constraint_scope.md').write_text('\n'.join(lines))
    if digest(folder / 'summary.json') != primary_sha:
        raise RuntimeError('Primary screening summary changed')
    return folder / 'region_constraint_scope.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(report(args.folder, args.plan))
