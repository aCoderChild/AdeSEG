"""Frozen native-schedule value consolidation study; Drive outputs only."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from evaluation.statistics import paired_wilcoxon,holm_adjust,sequence_bootstrap_interval
from modeling.native_value_consolidation import NativeValueConsolidation
from scripts.memory_token_io import digest,validate_output
from scripts.run_regional_refinement import write
from scripts.run_soft_role_affinity_cv import summarize
from scripts.run_region_constrained_tome_gate import native_reference
from scripts.run_wider_history_tome_gate import run_condition,verify_native_setup,NATIVE_RESULTS,NATIVE_PROTOCOL

SEQUENCES=tuple(f'seq{i}' for i in range(2,24) if i!=7)
CONTRASTS=(('value_minus_native','value','native'),('value_minus_joint','value','joint'),('joint_minus_native','joint','native'))


def compare(summaries):
    if set(summaries)!=set(('native','value','joint')) or any(set(x)!=set(SEQUENCES) for x in summaries.values()):
        raise ValueError('Require all 21 eligible sequences in all arms')
    contrasts={}
    for name,left,right in CONTRASTS:
        a={s:summaries[left][s]['present_dice'] for s in SEQUENCES}
        b={s:summaries[right][s]['present_dice'] for s in SEQUENCES}
        delta={s:a[s]-b[s] for s in SEQUENCES}
        contrasts[name]={**paired_wilcoxon(a,b),'mean_delta':float(np.mean(list(delta.values()))),
          'ci95':sequence_bootstrap_interval(delta,seed=20261112),
          'win_tie_loss':[sum(x>1e-12 for x in delta.values()),sum(abs(x)<=1e-12 for x in delta.values()),sum(x< -1e-12 for x in delta.values())],
          'deltas':delta,'mean_delta_without_seq8':float(np.mean([x for s,x in delta.items() if s!='seq8']))}
    adjusted=holm_adjust({n:x['p'] for n,x in contrasts.items()})
    for n,x in contrasts.items():x['holm_p']=adjusted[n]
    cohort={}
    for arm,seqs in summaries.items():
        cohort[arm]={k:float(np.mean([x[k] for x in seqs.values()])) for k in ['present_dice','iou','boundary_f']}
        for k in ['gt_empty_false_alarm_count','gt_empty_targets','return_count','undetected_returns']:
            cohort[arm][k]=sum(x[k] for x in seqs.values())
    primary,control=contrasts['value_minus_native'],contrasts['value_minus_joint']
    checks={'gain_vs_native_at_least_0_01':primary['mean_delta']>=0.01,
      'gain_vs_joint_at_least_0_002':control['mean_delta']>=0.002,
      'native_ci_lower_positive':primary['ci95'][0]>0,'control_ci_lower_positive':control['ci95'][0]>0,
      'native_holm_p_below_0_05':primary['holm_p']<0.05,'control_holm_p_below_0_05':control['holm_p']<0.05,
      'at_least_14_native_wins':primary['win_tie_loss'][0]>=14,
      'gain_without_seq8_positive':primary['mean_delta_without_seq8']>0,
      'control_gain_without_seq8_positive':control['mean_delta_without_seq8']>0,
      'no_boundary_f_loss':cohort['value']['boundary_f']>=cohort['native']['boundary_f'],
      'no_more_false_alarms':cohort['value']['gt_empty_false_alarm_count']<=cohort['native']['gt_empty_false_alarm_count'],
      'no_more_undetected_returns':cohort['value']['undetected_returns']<=cohort['native']['undetected_returns']}
    return {'cohort':cohort,'contrasts':contrasts,'screen':{**checks,'passed':all(checks.values())},
      'stretch_target_0_74':cohort['value']['present_dice']>=0.74,
      'interpretation':'Exploratory full-video result; patient independence unverified. Framewise competitiveness not established from unmatched 0.74. Untouched confirmation required.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();p=json.loads(args.protocol.read_text())
    if p['status']!='frozen_for_full_cohort' or digest(args.protocol)!=args.protocol.with_suffix('.json.sha256').read_text().strip():raise RuntimeError('Protocol differs')
    if tuple(p['sequences'])!=SEQUENCES:raise RuntimeError('Cohort differs')
    for n,h in p['source_hashes'].items():
        if digest(ROOT/n)!=h:raise RuntimeError('Source changed: '+n)
    for n,h in p['input_hashes'].items():
        if digest(Path(n))!=h:raise RuntimeError('Input changed: '+n)
    gate=Path(p['gate_directory']);done=json.loads((gate/'completion.json').read_text())
    if not done['passed'] or not done['fresh_process_results_equal']:raise RuntimeError('Propagation gate incomplete')
    out=validate_output(args.output_dir)
    if out.exists():raise RuntimeError('Use a fresh output directory')
    out.mkdir();write(out/'setup.json',p)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    checkpoint=Path(p['checkpoint']);manifest=ROOT/'data/polypgen_sequence.jsonl';verify_native_setup(checkpoint,manifest)
    videos={**load_manifest(manifest,'dev'),**load_manifest(manifest,'test')}
    native_results=json.loads(NATIVE_RESULTS.read_text());native_protocol=json.loads(NATIVE_PROTOCOL.read_text())
    summaries={a:{} for a in ['native','value','joint']};rows={a:{} for a in ['native','value','joint']};runtime={a:{} for a in ['value','joint']}
    try:
        for seq in SEQUENCES:
            prompt,native,receipt=native_reference(NATIVE_RESULTS.parent,native_results,native_protocol,seq,videos[seq])
            rows['native'][seq]=list(native.values());summaries['native'][seq]=summarize(rows['native'][seq])
            for mode in ['value','joint']:
                write(out/'phase.json',{'sequence':seq,'mode':mode,'complete_sequences':len(summaries['joint'])})
                current,rt=run_condition('global',seq,videos[seq],prompt,native,checkpoint,out,
                    max_noncond_maps=6,allocation='uniform',condition_name=mode,
                    memory_reader_factory=lambda model:NativeValueConsolidation(model,mode=mode),native_exact_required=False)
                if [x['frame'] for x in current]!=list(native):raise RuntimeError('Frame coverage differs')
                if any(x['present']!=native[x['frame']]['present'] for x in current):raise RuntimeError('GT presence differs')
                rows[mode][seq]=current;summaries[mode][seq]=summarize(current);runtime[mode][seq]=rt
                write(out/'sequence_summaries.json',summaries);write(out/'runtime.json',runtime)
            for x,y in zip(rows['value'][seq],rows['joint'][seq],strict=True):
                for a,b in zip(x['attention_reads'],y['attention_reads'],strict=True):
                    if any(a[k]!=b[k] for k in ['frames','pointer_keys','spatial_keys_before','spatial_keys_after']):raise RuntimeError('Matched source schedule differs')
        result={'sequence_summaries':summaries,'runtime':runtime,'comparison':compare(summaries),'frame_rows':rows}
        for n,h in p['source_hashes'].items():
            if digest(ROOT/n)!=h:raise RuntimeError('Source changed during inference: '+n)
        write(out/'results.json',result);write(out/'completion.json',{'complete':True,'sequences':list(SEQUENCES),'screen_passed':result['comparison']['screen']['passed'],'results_sha256':digest(out/'results.json')})
        write(out/'progress.json',{'status':'complete','sequences':21,'screen_passed':result['comparison']['screen']['passed']})
    except Exception as exc:
        write(out/'failure.json',{'error':str(exc),'type':type(exc).__name__});raise


if __name__=='__main__':main()
