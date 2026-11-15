#!/usr/bin/env python3
"""Compare verified matched selectors at two post-hoc retention capacities."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('small','large','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();seqs=('seq8','seq11','seq20','seq23');arms=('usage','contrast','distributed','shuffled_contrast','rgb_contrast','prompt_conditioned','descriptor_independent','mismatched_prompt');lines=['# Fixed-read comparison across two history capacities','','Both capacities use the same frozen read, prompt and source tokens. The first retains 4,096 of 5,120 eligible historical tokens; the second retains 1,024. The second capacity was chosen after inspecting the first result, so this is exploratory. All methods within a capacity have the same attention-key count; the full-read no-op is an unequal-key reference. Means cover only the selected present reads.','','| Sequence | Present / empty | Capacity | No-op Dice | Usage Dice | Unweighted Dice | Distributed Dice | Shuffled Dice | RGB Dice | Trained prompt Dice | Empty false alarms, trained |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for seq in seqs:
 pair=[]
 for root in (a.large,a.small):
  audit=root/f'{seq}_audit_attempt01';result=read(audit/'results.json');completion=read(audit/'completion.json');vr=read(root/f'{seq}_audit_verification_attempt01'/'verification.json');rank=read(root/f'{seq}_rank_verification_attempt01'/'verification.json');replay=read(root/f'{seq}_replay_attempt01'/'completion.json')
  if not all(x['passed'] for x in (completion,vr,rank,replay)) or completion['results_sha256']!=digest(audit/'results.json') or vr['audit_sha256']!=completion['results_sha256'] or rank['replay_sha256']!=digest(root/f'{seq}_replay_attempt01'/'completion.json'):raise ValueError('Missing or altered verification')
  pair.append((result,replay))
 x,y=pair
 if [(r['frame'],r['label_sha256'],r['present']) for r in x[0]['frames']]!=[(r['frame'],r['label_sha256'],r['present']) for r in y[0]['frames']]:raise ValueError('Frames or labels not matched')
 if [(r['frame'],r['masks']['noop']) for r in x[1]['events']]!=[(r['frame'],r['masks']['noop']) for r in y[1]['events']]:raise ValueError('Full-read reference changed')
 for capacity,(result,_) in zip((4096,1024),pair):
  pos=[r for r in result['frames'] if r['present']];empty=[r for r in result['frames'] if not r['present']];means=[]
  for m in ('noop','usage','contrast','distributed','shuffled_contrast','rgb_contrast','prompt_conditioned'):
   means.append(f"{np.mean([r['scores'][m]['dice'] for r in pos]):.6f}" if pos else '—')
  alarms=sum(bool(r['scores']['prompt_conditioned']['false_alarm']) for r in empty)
  lines.append(f"| {seq} | {len(pos)} / {len(empty)} | {capacity} | "+' | '.join(means)+f' | {alarms} / {len(empty)} |')
lines+=['','These are local read interventions, not persistent token banks, independent patient validation or full-video inference. All selected frames and both capacities have been inspected; no capacity or method has been confirmed on a held-out cohort. Sequence 23 uses the fixed inset-free viewport.','']
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'report_sha256':digest(out/'report.md'),'runner_sha256':digest(Path(__file__))},indent=2)+'\n');print('\n'.join(lines))
