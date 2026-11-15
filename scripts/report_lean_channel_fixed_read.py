#!/usr/bin/env python3
"""Summarize only independently verified local token-deletion experiments."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('root','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();methods=['noop','usage','contrast','distributed','three_region','foreground','shuffled','rgb_contrast','prompt_conditioned','descriptor_independent','mismatched_prompt'];data={};events={}
for seq in ('seq11','seq23'):
 audit=a.root/f'{seq}_audit_attempt01';v=a.root/f'{seq}_audit_verification_attempt01';replay=a.root/f'{seq}_replay_attempt01'
 check=read(v/'verification.json');done=read(audit/'completion.json');d=read(audit/'results.json');r=read(replay/'completion.json')
 if not check['passed'] or check['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_channel_priority_audit.py') or check['audit_sha256']!=digest(audit/'results.json') or not done['passed'] or done['results_sha256']!=digest(audit/'results.json') or set(d['frames'][0]['scores'])!=set(methods):raise ValueError('Audit incomplete')
 data[seq]=d;events[seq]=r['events']
lines=['# Trained channel attention: matched fixed-read deletion experiment','','Eight previously selected, exploratory reads from sequences 11 and 23 were replayed. At each read, exactly 1,024 eligible historical tokens were deleted according to each ranking; prompt memory, newest memory, pointer tokens and other entries remained intact. The ranks determine which high-ranked tokens are removed, not which tokens are retained. This is a sensitivity test: a decrease in Dice after deletion may suggest selected tokens contributed at that read, whereas an increase does not prove a selector would improve video propagation. It is not a direct fixed-budget retention experiment.','','Two independent feature reconstructions and two independent replay processes produced exact matching outputs. Unweighted FPN contrast reproduced the originally captured source scores exactly at every source token; previous unmodified control masks and no-op masks also matched. Labels were opened after replay. Sequence 23 used its inset-free viewport.','','| Sequence | Present reads | Method | Mean present Dice | Empty false alarms / empty reads |','| --- | ---: | --- | ---: | ---: |']
for seq in ('seq11','seq23'):
 rows=data[seq]['frames'];positive=[r for r in rows if r['present']];empty=[r for r in rows if not r['present']]
 for method in methods:
  dice=np.mean([r['scores'][method]['dice'] for r in positive]) if positive else float('nan');false=sum(bool(r['scores'][method]['false_alarm']) for r in empty)
  lines.append(f'| {seq} | {len(positive)} | {method} | {dice:.6f} | {false} / {len(empty)} |')
lines+=['','## Deletion-set overlap','','Mean Jaccard overlap between the 1,024 tokens deleted by the prompt-conditioned arm and by each comparator, across the four reads in each sequence. High overlap can explain similar outcomes; it does not by itself prove that a scoring signal is useful.','','| Sequence | Unweighted contrast | Usage | Distributed | Descriptor-independent | Mismatched descriptor |','| --- | ---: | ---: | ---: | ---: | ---: |']
for seq in ('seq11','seq23'):
 row=[]
 for method in ('contrast','usage','distributed','descriptor_independent','mismatched_prompt'):
  values=[]
  for e in events[seq]:
   a0=set(e['deleted']['prompt_conditioned']);b0=set(e['deleted'][method]);values.append(len(a0&b0)/len(a0|b0))
  row.append(np.mean(values))
 lines.append('| '+seq+' | '+' | '.join(f'{x:.4f}' for x in row)+' |')
lines+=['','The gate was trained on 15 frames from three other sequences, and its small sequence-separated feature-ranking validation did not outperform unweighted FPN. Patient identity is unknown. The eight reads were previously inspected and are not independent confirmation, do not cover sequences 8 or 20, and do not measure full-video propagation, runtime, or causal effects under a different memory state. No result here establishes a deployable improvement.','']
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'report_sha256':digest(out/'report.md'),'audits':{seq:digest(a.root/f'{seq}_audit_attempt01/results.json') for seq in data},'generator_sha256':digest(Path(__file__))},indent=2)+'\n');print('\n'.join(lines))
