#!/usr/bin/env python3
"""Compare frozen historical-usage and control choices on audited masks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('usage','source-deletions','random-deletions','loso-audit','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());la=read(a.loso_audit/'results.json');ld=read(a.loso_audit/'completion.json')
if not ld['passed'] or not la['passed'] or ld['results_sha256']!=digest(a.loso_audit/'results.json') or la['runner_sha256']!=digest(ROOT/'scripts/audit_lean_source_utility_loso.py'):raise ValueError('LOSO audit changed')
arms=('historical_usage','utility_trained','unweighted','rgb','shuffled','age');rows=[];details=[]
for seq in ('seq2','seq3','seq4','seq9','seq10'):
 xfile=a.usage/f'{seq}_scores_attempt03/scores.json';yfile=a.usage/f'{seq}_scores_attempt04/scores.json';x=read(xfile);y=read(yfile)
 if x!=y or not x['passed'] or x['sequence']!=seq or x['loso_predictions_sha256']!=la['fold_sha256'][seq] or x['loso_audit_completion_sha256']!=digest(a.loso_audit/'completion.json') or x['runner_sha256']!=digest(ROOT/'scripts/score_lean_historical_usage_groups.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_historical_usage_group_control_v1.json') or x['percentile_sha256']!=digest(ROOT/'modeling/lean_historical_usage_score.py'):raise ValueError('Frozen historical usage scores changed')
 outcomes=[]
 for root,verifier in ((a.source_deletions,'verify_lean_source_group_scores.py'),(a.random_deletions,'verify_lean_utility_group_scores.py')):
  folder=root/f'{seq}_audit_attempt01';verify=root/f'{seq}_score_verification_attempt01';data=read(folder/'results.json');v=read(verify/'verification.json')
  if not v['passed'] or v['results_sha256']!=digest(folder/'results.json') or v['verifier_sha256']!=digest(ROOT/'scripts'/verifier):raise ValueError('Audited outcomes changed')
  replay=root/('gate_attempt01' if seq=='seq3' else f'{seq}_attempt01');r=read(replay/'completion.json')
  if data['replay_sha256']!=digest(replay/'completion.json'):raise ValueError('Replay changed')
  outcomes.append((data,{z['frame']:z for z in r['events']}))
 (source,srep),(random,rrep)=outcomes
 if x['source_replay_sha256']!=source['replay_sha256']:raise ValueError('Source scores and outcomes differ')
 left={f['frame']:f for f in source['frames']};right={f['frame']:f for f in random['frames']};scores={f['frame']:f['scores'] for f in x['events']}
 if set(left)!=set(right) or set(left)!=set(scores):raise ValueError('Matched read coverage differs')
 per={k:[] for k in arms};rank={k:[] for k in arms};fa={k:0 for k in arms};source_avg=[];dist_avg=[];empty=0;dist_empty=0.
 for frame in sorted(left):
  s=left[frame];r=right[frame];se=srep[frame];re=rrep[frame]
  if s['present']!=r['present'] or s['label_sha256']!=r['label_sha256'] or s['scores']['noop']!=r['scores']['noop'] or se['key_counts']!=re['key_counts'] or {j for g in se['groups'] for j in g}!={j for g in re['groups'] for j in g}:raise ValueError('Read, baseline or key budget differs')
  q=scores[frame]
  if any(len(q[k])!=5 or not np.isfinite(q[k]).all() for k in arms):raise ValueError('Invalid score vector')
  choices={k:min(range(5),key=lambda i:(q[k][i],i)) for k in arms};sm=[s['scores'][f'group{i}'] for i in range(5)];rm=[r['scores'][f'group{i}'] for i in range(5)]
  if s['present']:
   avg=sum(z['dice'] for z in sm)/5;davg=sum(z['dice'] for z in rm)/5;source_avg.append(avg);dist_avg.append(davg);utility=[s['scores']['noop']['dice']-z['dice'] for z in sm]
   for k in arms:
    per[k].append(sm[choices[k]]['dice']-avg);pairs=[]
    for i in range(5):
     for j in range(i+1,5):
      du=utility[i]-utility[j]
      if du==0:continue
      prod=(q[k][i]-q[k][j])*du;pairs.append(1. if prod>0 else 0. if prod<0 else .5)
    if pairs:rank[k].append(float(np.mean(pairs)))
  else:
   empty+=1;dist_empty+=sum(bool(z['false_alarm']) for z in rm)/5
   for k in arms:fa[k]+=int(sm[choices[k]]['false_alarm'])
  details.append({'sequence':seq,'frame':frame,'present':s['present'],'selected_groups':choices})
 rows.append({'sequence':seq,'present_reads':len(source_avg),'empty_reads':empty,'rank_concordance':{k:float(np.mean(rank[k])) if rank[k] else None for k in arms},'dice_minus_source_average':{k:float(np.mean(per[k])) if per[k] else None for k in arms},'source_average_dice':float(np.mean(source_avg)) if source_avg else None,'distributed_random_average_dice':float(np.mean(dist_avg)) if dist_avg else None,'empty_false_alarms':fa,'distributed_random_expected_empty_false_alarms':dist_empty,'scores_sha256':digest(xfile),'repeated_scores_sha256':digest(yfile),'source_outcome_sha256':digest(a.source_deletions/f'{seq}_audit_attempt01/results.json'),'random_outcome_sha256':digest(a.random_deletions/f'{seq}_audit_attempt01/results.json')})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'rows':rows,'details':details,'protocol_sha256':digest(ROOT/'protocols/lean_historical_usage_group_control_v1.json'),'runner_sha256':digest(Path(__file__))};(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Historical Lean usage: matched source-deletion control','','Each method removes one 1,024-key source using a label-free score. Historical usage was measured when that source was newest, before the target read; it is not attention from the evaluated target. Δ Dice is relative to the mean of the five same-read source deletions. The distributed comparison averages five previously frozen mixed-source deletions at the same key count.','','| Sequence | Present / empty | Usage rank | Trained gate rank | RGB rank | Usage Δ Dice | Gate Δ Dice | Age Δ Dice | Source average Dice | Distributed random average Dice |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 f=lambda x:'—' if x is None else f'{x:.3f}';g=lambda x:'—' if x is None else f'{x:+.7f}';h=lambda x:'—' if x is None else f'{x:.7f}';c=r['rank_concordance'];d=r['dice_minus_source_average'];lines.append(f"| {r['sequence']} | {r['present_reads']} / {r['empty_reads']} | {f(c['historical_usage'])} | {f(c['utility_trained'])} | {f(c['rgb'])} | {g(d['historical_usage'])} | {g(d['utility_trained'])} | {g(d['age'])} | {h(r['source_average_dice'])} | {h(r['distributed_random_average_dice'])} |")
lines+=['','Scores and source mappings were frozen and repeated before joining independently checked masks. The five source arms and five mixed-source arms share the same key universe and budget, but their results are correlated; random-distributed is an average, not a trained online policy. The study and sequences were previously inspected, patient independence is unknown, and local deletion effects are small and conditional. One eligible empty read is insufficient to assess false alarms; exact counts are in results.json. No selected mask was propagated.']
(out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
