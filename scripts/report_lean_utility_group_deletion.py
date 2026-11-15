#!/usr/bin/env python3
"""Summarize verified sequence-separated group-deletion scores."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();read=lambda f:json.loads(f.read_text());data=[]
for seq in ('seq2','seq3','seq4','seq9','seq10'):
 folder=a.source/f'{seq}_audit_attempt01';verify=a.source/f'{seq}_score_verification_attempt01';r=read(folder/'results.json');v=read(verify/'verification.json')
 if not v['passed'] or v['results_sha256']!=digest(folder/'results.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_utility_group_scores.py') or r['sequence']!=seq:raise ValueError('Unverified result')
 delta=[];false_changes=0;present=0;empty=0
 for row in r['frames']:
  base=row['scores']['noop'];present+=int(row['present']);empty+=int(not row['present'])
  for i in range(5):
   arm=row['scores'][f'group{i}']
   if row['present']:delta.append(arm['dice']-base['dice'])
   else:false_changes+=int(arm['false_alarm']!=base['false_alarm'])
 vals=np.abs(np.asarray(delta,dtype=float));data.append({'sequence':seq,'split':'development' if seq in ('seq2','seq3','seq4') else 'validation','present_reads':present,'empty_reads':empty,'excluded_reads':r['excluded_reads'],'present_comparisons':len(delta),'median_absolute_dice_change':float(np.median(vals)) if len(vals) else None,'max_absolute_dice_change':float(np.max(vals)) if len(vals) else None,'empty_comparisons':5*empty,'empty_false_alarm_changes':false_changes,'results_sha256':digest(folder/'results.json'),'score_verification_sha256':digest(verify/'verification.json')})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'sequences':data,'protocol_sha256':digest(ROOT/'protocols/lean_utility_group_deletion_v1.json'),'runner_sha256':digest(Path(__file__))};(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Sequence-separated fixed-read group-deletion study','','Five disjoint groups of 1,024 historical keys were deleted one at a time from each eligible frozen read. Every deletion arm has the same key count. The full-read no-op has 1,024 extra keys and is a descriptive reference. No group was selected using target labels.','','| Sequence | Split | Present reads | Empty reads | Excluded | Median absolute Dice change | Maximum absolute Dice change | Empty-frame alarm changes |','|---|---|---:|---:|---:|---:|---:|---:|']
for d in data:
 med=f"{d['median_absolute_dice_change']:.7f}" if d['median_absolute_dice_change'] is not None else '—';mx=f"{d['max_absolute_dice_change']:.7f}" if d['max_absolute_dice_change'] is not None else '—';lines.append(f"| {d['sequence']} | {d['split']} | {d['present_reads']} | {d['empty_reads']} | {d['excluded_reads']} | {med} | {mx} | {d['empty_false_alarm_changes']} / {d['empty_comparisons']} |")
lines+=['','The magnitudes are small for these randomly partitioned groups. This makes the proposed deletion-derived utility target difficult to distinguish from weak, conditional effects; it does not prove that individual tokens or other grouping schemes are unhelpful. Groups share reads and are not independent samples. The one eligible empty read is insufficient to assess false-alarm safety. Previously examined data and sequence-level splitting do not establish patient independence. No model was trained, and these local replays do not establish a full-video improvement. Source early-feature descriptors, inference-time usage controls, and a verified training target are still needed before a trained selector can be evaluated.']
(out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
