#!/usr/bin/env python3
"""Compare verified source-block and random 1024-key deletion effects."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('random','source','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());stats=[]
for seq in ('seq2','seq3','seq4','seq9','seq10'):
 datasets=[]
 for root,verifier in ((a.random,'verify_lean_utility_group_scores.py'),(a.source,'verify_lean_source_group_scores.py')):
  audit=root/f'{seq}_audit_attempt01';check=root/f'{seq}_score_verification_attempt01';scores=read(audit/'results.json');v=read(check/'verification.json')
  if not v['passed'] or v['results_sha256']!=digest(audit/'results.json') or v['verifier_sha256']!=digest(ROOT/'scripts'/verifier):raise ValueError('Score audit invalid')
  replay=root/('gate_attempt01' if seq=='seq3' else f'{seq}_attempt01');run=read(replay/'completion.json')
  if scores['replay_sha256']!=digest(replay/'completion.json'):raise ValueError('Replay changed')
  datasets.append((scores,{e['frame']:e for e in run['events']}))
 (rnd,re),(src,se)=datasets
 rf={e['frame']:e for e in rnd['frames']};sf={e['frame']:e for e in src['frames']};common=sorted(set(rf)&set(sf));deltas=[[],[]];per_read=[[],[]];empty=0;changes=[0,0]
 for frame in common:
  rr,sr=rf[frame],sf[frame];a0=re[frame];b0=se[frame]
  if rr['label_sha256']!=sr['label_sha256'] or rr['present']!=sr['present'] or rr['scores']['noop']!=sr['scores']['noop'] or a0['key_counts']!=b0['key_counts'] or {x for g in a0['groups'] for x in g}!={x for g in b0['groups'] for x in g}:raise ValueError('Read, no-op, budget or eligible membership differs')
  if rr['present']:
   for i,row in enumerate((rr,sr)):
    vals=[row['scores'][f'group{j}']['dice']-row['scores']['noop']['dice'] for j in range(5)]
    deltas[i].extend(abs(x) for x in vals);per_read[i].append(max(abs(x) for x in vals))
  else:
   empty+=1
   for i,row in enumerate((rr,sr)):changes[i]+=sum(row['scores'][f'group{j}']['false_alarm']!=row['scores']['noop']['false_alarm'] for j in range(5))
 d={'sequence':seq,'split':'development' if seq in ('seq2','seq3','seq4') else 'validation','matched_reads':len(common),'matched_present':len(per_read[0]),'matched_empty':empty,'random_only_reads':len(set(rf)-set(sf)),'source_only_reads':len(set(sf)-set(rf)),'random_median_abs':float(np.median(deltas[0])) if deltas[0] else None,'source_median_abs':float(np.median(deltas[1])) if deltas[1] else None,'random_max_abs':float(max(deltas[0])) if deltas[0] else None,'source_max_abs':float(max(deltas[1])) if deltas[1] else None,'source_larger_read_max':sum(x>y for x,y in zip(per_read[1],per_read[0])),'empty_alarm_changes_random':changes[0],'empty_alarm_changes_source':changes[1]};stats.append(d)
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'results.json').write_text(json.dumps({'rows':stats,'runner_sha256':digest(Path(__file__))},indent=2)+'\n')
lines=['# Complete-source versus random token-group deletions','','Each arm removes 1,024 keys from the same frozen read. Random and source arms use the same eligible keys and have the same key count; comparisons to the full-read no-op are not matched-key comparisons. This follow-up was chosen after observing the random results.','','| Sequence | Split | Present / empty reads | Median abs Dice change: random / source | Max abs Dice change: random / source | Source has larger per-read maximum |','|---|---|---:|---:|---:|---:|']
for d in stats:
 fmt=lambda x: '—' if x is None else f'{x:.6f}'
 lines.append(f"| {d['sequence']} | {d['split']} | {d['matched_present']} / {d['matched_empty']} | {fmt(d['random_median_abs'])} / {fmt(d['source_median_abs'])} | {fmt(d['random_max_abs'])} / {fmt(d['source_max_abs'])} | {d['source_larger_read_max']} / {d['matched_present']} |")
lines+=['','All reads and arms are dependent, and split is by sequence with patient independence unverified. The empty-frame sample is very small. These are conditional, local interventions, not full-video propagation or proof of token-level utility. The design was selected after inspecting previous random-group results and should not be interpreted as an untouched confirmatory test.']
(out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
