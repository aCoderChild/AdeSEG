"""Sequence-unit value-repair comparison using existing statistical routines."""
import math
import numpy as np
from evaluation.statistics import paired_wilcoxon,sequence_bootstrap_interval,holm_adjust

ARMS=('native','segmentation_only','segmentation_plus_consistency')
METRICS=('present_dice','present_iou','present_boundary_f','gt_empty_false_alarm_rate')


def clean_json(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    return value


def summarize(groups,names,seed,apply_validation_screen=False):
    if set(groups)!=set(ARMS) or not names or any(set(groups[a])!=set(names) for a in ARMS):
        raise ValueError('Missing/extra arm or sequence in matched comparison')
    for name in names:
        base=groups['native'][name]
        for arm in ARMS:
            row=groups[arm][name]
            if row.get('sequence')!=name or row['present_frames']+row['gt_empty_frames']!=row['target_frames']:
                raise ValueError('Sequence ownership or labelled frame denominators differ')
            if any(row[k]!=base[k] for k in ('target_frames','present_frames','gt_empty_frames','prompt_frame')):
                raise ValueError('Methods do not cover the same labelled frame schedule')
            events=[(e['return_frame'],e['followup_sampled_frames']) for e in row['propagation']['returns']]
            if events!=[(e['return_frame'],e['followup_sampled_frames']) for e in base['propagation']['returns']]:
                raise ValueError('Methods have different GT-defined return events')
            if any(row[m] is not None and (not math.isfinite(row[m]) or not 0<=row[m]<=1) for m in METRICS):
                raise ValueError('Nonfinite or invalid sequence metric')
    means={a:{m:float(np.mean([groups[a][n][m] for n in names if groups[a][n][m] is not None]))
        if any(groups[a][n][m] is not None for n in names) else None for m in METRICS} for a in ARMS}
    contrasts={}
    for control in ['native','segmentation_only']:
        eligible=[n for n in names if groups['segmentation_plus_consistency'][n]['present_dice'] is not None and groups[control][n]['present_dice'] is not None]
        left={n:groups['segmentation_plus_consistency'][n]['present_dice'] for n in eligible};right={n:groups[control][n]['present_dice'] for n in eligible}
        delta={n:left[n]-right[n] for n in eligible};vals=list(delta.values())
        contrasts[control]={'n_sequences':len(eligible),'sequence_delta':delta,'mean_delta':float(np.mean(vals)) if vals else None,
            'ci95':sequence_bootstrap_interval(delta,seed=seed) if vals else None,'wilcoxon':paired_wilcoxon(left,right),
            'wins':sum(d>0 for d in vals),'ties':sum(d==0 for d in vals),'losses':sum(d<0 for d in vals),
            'leave_one_sequence_out':{n:float(np.mean([d for k,d in delta.items() if k!=n])) for n in eligible} if len(eligible)>1 else {}}
    adjusted=holm_adjust({k:v['wilcoxon']['p'] for k,v in contrasts.items()})
    for k in contrasts:contrasts[k]['holm_p']=adjusted[k]
    temporal={}
    for arm in ARMS:
        events=[e for n in names for e in groups[arm][n]['propagation']['returns']];delays=[e['delay_sampled_frames'] for e in events if e['detected']]
        empty=sum(groups[arm][n]['gt_empty_frames'] for n in names);fa=sum(groups[arm][n]['gt_empty_false_alarm_count'] for n in names)
        seconds=sum(groups[arm][n]['elapsed_seconds'] for n in names);targets=sum(groups[arm][n]['target_frames'] for n in names)
        temporal[arm]={'return_events':len(events),'detected_returns':sum(e['detected'] for e in events),
            'missed_returns':sum(not e['detected'] for e in events),'conditional_mean_delay_sampled_frames':float(np.mean(delays)) if delays else None,
            'gt_empty_false_alarms':fa,'gt_empty_frames':empty,'pooled_false_alarm_rate':fa/empty if empty else None,
            'instrumented_elapsed_seconds':seconds,'instrumented_fps':targets/seconds if seconds else None}
    screen=None
    if apply_validation_screen:
        if names!=['seq13','seq14','seq15']:raise ValueError('Release screen is fixed to validation seq13-15')
        native,matched=contrasts['native'],contrasts['segmentation_only'];fa0=means['native']['gt_empty_false_alarm_rate'];fa2=means['segmentation_plus_consistency']['gt_empty_false_alarm_rate']
        fa_delta=fa2-fa0 if fa0 is not None and fa2 is not None else None
        conditions={'all_three_present_sequences':native['n_sequences']==3 and matched['n_sequences']==3,
            'gain_native_at_least_0_01':native['mean_delta'] is not None and native['mean_delta']>=.01,
            'gain_matched_at_least_0_005':matched['mean_delta'] is not None and matched['mean_delta']>=.005,
            'wins_native_at_least_2':native['wins']>=2,'wins_matched_at_least_2':matched['wins']>=2,
            'false_alarm_increase_at_most_0_02':fa_delta is not None and fa_delta<=.02}
        screen={'passed':all(conditions.values()),'conditions':conditions,'false_alarm_equal_sequence_delta':fa_delta,
            'interpretation':'Exploratory release screen, not statistical significance or independent confirmation'}
    return clean_json({'per_sequence':groups,'equal_sequence_means':means,'primary_contrasts_C2_minus':contrasts,
        'temporal_and_compute':temporal,'validation_screen':screen,'statistical_unit':'sequence; two primary contrasts Holm-corrected',
        'limits':'Conditional re-detection delay excludes missed returns and is descriptive. FPS includes integrity instrumentation; no controlled speed claim.'})
