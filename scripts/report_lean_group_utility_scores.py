#!/usr/bin/env python3
"""Compare frozen label-free group scores with independently audited local effects."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('scores','deletions','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());rows=[];details=[];arms=('cbam','unweighted','rgb','shuffled','age')
for seq in ('seq2','seq3','seq4','seq9','seq10'):
 xpath=a.scores/f'{seq}_scores_attempt02/scores.json';ypath=a.scores/f'{seq}_scores_attempt03/scores.json';x=read(xpath);y=read(ypath)
 audit=a.deletions/f'{seq}_audit_attempt01';check=a.deletions/f'{seq}_score_verification_attempt01';r=read(audit/'results.json');v=read(check/'verification.json')
 if x!=y or not x['passed'] or x['sequence']!=seq or x['runner_sha256']!=digest(ROOT/'scripts/score_lean_group_utility_features.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_group_utility_score_diagnostic_v1.json') or not v['passed'] or v['results_sha256']!=digest(audit/'results.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_utility_group_scores.py') or r['sequence']!=seq or x['replay_sha256']!=r['replay_sha256']:
  raise ValueError('Scores, outcomes or provenance differ')
 lookup={z['frame']:z for z in r['frames']};per_arm={k:[] for k in arms};empty=0;utility_ties=0
 if set(lookup)!={e['frame'] for e in x['events']}:raise ValueError('Frame coverage differs')
 for event in x['events']:
  row=lookup[event['frame']]
  if not row['present']:empty+=1;continue
  utility=[row['scores']['noop']['dice']-row['scores'][f'group{i}']['dice'] for i in range(5)];per_read={}
  for arm in arms:
   scores=event['scores'][arm]
   if len(scores)!=5 or not np.isfinite(scores).all():raise ValueError('Invalid frozen scores')
   values=[]
   for i in range(5):
    for j in range(i+1,5):
     diff=utility[i]-utility[j]
     if diff==0:
      if arm==arms[0]:utility_ties+=1
      continue
     product=(scores[i]-scores[j])*diff;values.append(1. if product>0 else 0. if product<0 else .5)
   if values:
    value=float(np.mean(values));per_arm[arm].append(value);per_read[arm]=value
  details.append({'sequence':seq,'frame':event['frame'],'utility':utility,'pairwise_concordance':per_read})
 rows.append({'sequence':seq,'split':'development' if seq in ('seq2','seq3','seq4') else 'validation','present_reads':len(per_arm['cbam']),'empty_reads':empty,'utility_tied_pairs':utility_ties,'concordance':{k:float(np.mean(vals)) if vals else None for k,vals in per_arm.items()},'score_sha256':digest(xpath),'repeated_score_sha256':digest(ypath),'outcome_sha256':digest(audit/'results.json')})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'rows':rows,'read_details':details,'runner_sha256':digest(Path(__file__)),'protocol_sha256':digest(ROOT/'protocols/lean_group_utility_score_diagnostic_v1.json')};(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Early-feature group scores versus local deletion effects','','Within each present read, pairwise concordance compares each group score with no-op Dice minus deletion Dice. 0.5 is chance for strict pairs; score ties receive half credit and tied utilities are excluded. The existing CBAM gate was trained on foreground occupancy, not utility. Scores were computed twice without reading deletion outcomes; the study itself was designed after prior outcomes had been inspected.','','| Sequence | Split | Present reads | CBAM | Unweighted | RGB | Shuffled | Age |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
 fmt=lambda n:'—' if n is None else f'{n:.3f}';c=r['concordance'];lines.append(f"| {r['sequence']} | {r['split']} | {r['present_reads']} | {fmt(c['cbam'])} | {fmt(c['unweighted'])} | {fmt(c['rgb'])} | {fmt(c['shuffled'])} | {fmt(c['age'])} |")
lines+=['','These random groups mix source frames and positions; small group-mean differences and very small deletion effects may not capture individual token utility. Comparisons share reads, the validation sequences were previously examined, and patient independence is unknown. The single eligible empty read cannot support a ranking or a false-alarm conclusion. Historical Lean usage scores were not captured in these data; current-read attention usage would require performing the full read first and is not presented as a pre-read baseline. Existing priority fixed-read experiments provide separate usage and distributed comparisons. No propagation or trained utility selector is evaluated here.']
(out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
