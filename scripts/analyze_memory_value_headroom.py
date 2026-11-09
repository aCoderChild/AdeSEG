"""Sequence-unit summaries for the fixed TRAIN-only, read-only oracle diagnostic."""
import numpy as np
from evaluation.statistics import sequence_bootstrap_interval, paired_wilcoxon, holm_adjust

ARMS = ('native', 'identity_values', 'mask_only', 'mask_and_presence')


def summarize(cases, seed):
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate oracle case')
    sequences = {}
    for name in sorted({c['sequence'] for c in cases}):
        rows = [c for c in cases if c['sequence'] == name]
        present = [c for c in rows if c['present']]
        absent = [c for c in rows if not c['present']]
        sequences[name] = {a: {
            'present_targets': len(present), 'empty_targets': len(absent),
            **{k: float(np.mean([c['arms'][a]['scores'][k] for c in present])) if present else None
               for k in ('dice', 'iou', 'boundary_f')},
            'false_alarms': sum(c['arms'][a]['prediction_nonempty'] for c in absent),
        } for a in ARMS}
    contrasts = {}
    native = {s: v['native']['dice'] for s, v in sequences.items() if v['native']['dice'] is not None}
    for a in ('mask_only', 'mask_and_presence'):
        treated = {s: sequences[s][a]['dice'] for s in native}
        delta = {s: treated[s] - native[s] for s in native}
        vals = list(delta.values())
        contrasts[a] = {'eligible_sequences': len(vals), 'per_sequence_delta_dice': delta,
            'mean_delta_dice': float(np.mean(vals)) if vals else None,
            'ci95': sequence_bootstrap_interval(delta, seed=seed, resamples=10000),
            'wilcoxon': paired_wilcoxon(treated, native),
            'wins': sum(v > 0 for v in vals), 'ties': sum(v == 0 for v in vals),
            'losses': sum(v < 0 for v in vals),
            'leave_one_out_mean': {s: float(np.mean([d for k,d in delta.items() if k != s])) for s in delta} if len(vals)>1 else {},
            'additional_false_alarm_cases': sum(not c['present'] and c['arms'][a]['prediction_nonempty']
                and not c['arms']['native']['prediction_nonempty'] for c in cases)}
    adjusted = holm_adjust({a: c['wilcoxon']['p'] for a,c in contrasts.items()})
    for a in contrasts: contrasts[a]['holm_p'] = adjusted[a]
    c = contrasts['mask_only']
    conditions = {'eight_present_sequences': c['eligible_sequences'] >= 8,
        'gain_at_least_0_01': c['mean_delta_dice'] is not None and c['mean_delta_dice'] >= .01,
        'ci_lower_above_zero': c['ci95'][0] > 0,
        'wins_at_least_60_percent': c['wins'] >= .6 * c['eligible_sequences'] and c['eligible_sequences'] > 0,
        'all_leave_one_out_positive': bool(c['leave_one_out_mean']) and all(v > 0 for v in c['leave_one_out_mean'].values()),
        'no_additional_false_alarm_cases': c['additional_false_alarm_cases'] == 0}
    empty = sum(not c['present'] for c in cases)
    return {'sequences': sequences, 'contrasts': contrasts, 'primary_decision_conditions': conditions,
        'primary_headroom_screen_passed': all(conditions.values()), 'empty_target_count': empty,
        'false_alarm_assessment': 'sampled_targets_only' if empty else 'unavailable_require_later_false_alarm_pilot',
        'redetection_assessment': 'not_estimable_from_sparse_read_only_interventions; no treated rollout',
        'efficacy_authorized': False, 'scope': 'TRAIN-only oracle headroom, not a learned segmentation result'}
