#!/usr/bin/env python3
"""Select frozen equal-budget deletions using label-free trained/control scores."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('training','training-audit','deletions','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());pred=read(a.training/'predictions.json');audit=read(a.training_audit/'results.json');done=read(a.training_audit/'completion.json')
if not pred['passed'] or pred['runner_sha256']!=digest(ROOT/'scripts/train_lean_source_utility_channel_pilot.py') or not audit['passed'] or not done['passed'] or done['results_sha256']!=digest(a.training_audit/'results.json') or audit['first_predictions_sha256']!=digest(a.training/'predictions.json') or audit['runner_sha256']!=digest(ROOT/'scripts/audit_lean_source_utility_channel_pilot.py'):raise ValueError('Trained model or audit changed')
preds={(r['sequence'],r['frame']):r for r in pred['predictions']};arms=('utility_trained','unweighted','cbam','rgb','shuffled','age');rows=[];details=[]
for seq in ('seq2','seq3','seq4','seq9','seq10'):
 folder=a.deletions/f'{seq}_audit_attempt01';ver=a.deletions/f'{seq}_score_verification_attempt01';data=read(folder/'results.json');v=read(ver/'verification.json')
 if not v['passed'] or v['results_sha256']!=digest(folder/'results.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_source_group_scores.py'):raise ValueError('Frozen outcome audit changed')
 present={k:[] for k in arms};delta={k:[] for k in arms};empty={k:0 for k in arms};expect=[];empty_expect=0.;nempty=0
 for frame in data['frames']:
  x=preds[(seq,frame['frame'])];scores={'utility_trained':x['scores'],**x['controls']}
  if any(len(scores[k])!=5 or not np.isfinite(scores[k]).all() for k in arms):raise ValueError('Invalid selection score')
  # Selection is determined only from frozen source descriptors and trained parameters.
  choices={k:min(range(5),key=lambda j:(scores[k][j],j)) for k in arms}
  masks=[frame['scores'][f'group{j}'] for j in range(5)]
  if frame['present']:
   mean=float(np.mean([m['dice'] for m in masks]));expect.append(mean)
   for arm in arms:
    value=masks[choices[arm]]['dice'];present[arm].append(value);delta[arm].append(value-mean)
  else:
   nempty+=1;empty_expect+=sum(bool(m['false_alarm']) for m in masks)/5
   for arm in arms:empty[arm]+=int(bool(masks[choices[arm]]['false_alarm']))
  details.append({'sequence':seq,'frame':frame['frame'],'present':frame['present'],'selected_groups':choices})
 rows.append({'sequence':seq,'split':'development' if seq in ('seq2','seq3','seq4') else 'validation','present_reads':len(expect),'empty_reads':nempty,'random_expected_present_dice':float(np.mean(expect)) if expect else None,'random_expected_empty_false_alarms':empty_expect,'methods':{k:{'present_dice':float(np.mean(present[k])) if present[k] else None,'paired_dice_minus_random':float(np.mean(delta[k])) if delta[k] else None,'empty_false_alarms':empty[k]} for k in arms},'outcome_sha256':digest(folder/'results.json')})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'rows':rows,'choices':details,'training_predictions_sha256':digest(a.training/'predictions.json'),'training_audit_sha256':digest(a.training_audit/'results.json'),'protocol_sha256':digest(ROOT/'protocols/lean_source_utility_group_selection_v1.json'),'runner_sha256':digest(Path(__file__))};(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Fixed-read selection from matched deletion arms','','Each method selects the lowest-scoring source to remove without using target labels. Every candidate removes 1,024 keys from the same read. The random comparison is the mean outcome over the five already computed groups. Differences are mean present-frame Dice minus that random expectation.','','| Sequence | Split | Present / empty reads | Utility-trained Δ Dice | Uniform Δ Dice | Occupancy-trained Δ Dice | RGB Δ Dice | Shuffled Δ Dice | Age Δ Dice |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 fmt=lambda v:'—' if v is None else f'{v:+.7f}';m=r['methods'];lines.append(f"| {r['sequence']} | {r['split']} | {r['present_reads']} / {r['empty_reads']} | "+' | '.join(fmt(m[k]['paired_dice_minus_random']) for k in arms)+' |')
lines+=['','The random expectation is descriptive, not an independently sampled video run. The evaluation reuses previously inspected data and outcomes; patient independence is unknown. All methods share reads and the local effects are very small. The single eligible empty read is insufficient to assess false alarms; per-method counts are in results.json. These selected frozen masks were not fed back into video memory and do not measure a propagated system. Pre-read Lean usage and a distributed policy were unavailable in this score set.']
(out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
