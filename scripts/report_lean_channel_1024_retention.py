#!/usr/bin/env python3
"""Create a per-sequence report from independently audited retention replays."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('root','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();methods=['noop','usage','contrast','distributed','three_region','foreground','shuffled_contrast','rgb_contrast','prompt_conditioned','descriptor_independent','mismatched_prompt'];data={};events={}
for seq in ('seq8','seq11','seq20','seq23'):
 audit=a.root/f'{seq}_audit_attempt01';verify=a.root/f'{seq}_audit_verification_attempt01';replay=a.root/f'{seq}_replay_attempt01'
 check=read(verify/'verification.json');done=read(audit/'completion.json');d=read(audit/'results.json');r=read(replay/'completion.json')
 if not check['passed'] or check['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_channel_1024_audit.py') or check['audit_sha256']!=digest(audit/'results.json') or not done['passed'] or done['results_sha256']!=digest(audit/'results.json') or any(set(row['scores'])!=set(methods) for row in d['frames']):raise ValueError('Audit incomplete')
 rank=read(a.root/f'{seq}_rank_verification_attempt01'/'verification.json')
 if not rank['passed'] or rank['rankings_checked']!=40 or rank['replay_sha256']!=digest(replay/'completion.json') or rank['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_1024_rankings.py'):raise ValueError('Retained ranking audit incomplete')
 data[seq]=d;events[seq]=r['events']
lines=['# Matched fixed-read retention of historical memory tokens','','This second retention budget was chosen after inspecting the first test and is post-hoc. Each sequence contributes four previously inspected read states. Each reduced arm retains exactly 1,024 of the same 5,120 eligible historical tokens; the remaining memory and object pointers are unchanged. All reduced arms have the same total attention-key count. The no-op uses 4,096 more keys and is a descriptive reference, not a matched-budget control. The shuffled arm permutes contrast independently within each source while leaving usage and token IDs fixed.','','The model receives the frozen original automatic prompt and memory state. Two fresh processes reproduced each result; an independent audit checked the retention complements, attention-key counts, rotary-position receipts and saved masks. Labels were used only after predictions were frozen. Sequence 23 used its fixed inset-excluded viewport.','','| Sequence | Present / empty reads | Method | Present-frame mean Dice | Empty-frame false alarms |','| --- | ---: | --- | ---: | ---: |']
for seq in data:
 rows=data[seq]['frames'];pos=[r for r in rows if r['present']];empty=[r for r in rows if not r['present']]
 for m in methods:
  value=f"{np.mean([r['scores'][m]['dice'] for r in pos]):.6f}" if pos else '—'
  lines.append(f"| {seq} | {len(pos)} / {len(empty)} | {m} | {value} | {sum(bool(r['scores'][m]['false_alarm']) for r in empty)} / {len(empty)} |")
lines+=['','## Retained-token overlap','','Mean Jaccard overlap of prompt-conditioned retained tokens with each alternative; high overlap may explain similar predictions but is not evidence of helpful token selection.','','| Sequence | Unweighted contrast | Usage | Distributed | Descriptor-independent | Mismatched descriptor |','| --- | ---: | ---: | ---: | ---: | ---: |']
for seq in data:
 values=[]
 for m in ('contrast','usage','distributed','descriptor_independent','mismatched_prompt'):
  vals=[]
  for e in events[seq]:
   x=set(e['kept']['prompt_conditioned']);y=set(e['kept'][m]);vals.append(len(x&y)/len(x|y))
  values.append(float(np.mean(vals)))
 lines.append('| '+seq+' | '+' | '.join(f'{x:.4f}' for x in values)+' |')
lines+=['','## Per-frame prompt-conditioned results','','| Sequence | Frame | Label present | No-op Dice | Usage Dice | Unweighted Dice | Prompt-conditioned Dice |','| --- | ---: | :---: | ---: | ---: | ---: | ---: |']
for seq in data:
 for row in data[seq]['frames']:
  vals=[row['scores'][m]['dice'] for m in ('noop','usage','contrast','prompt_conditioned')]
  lines.append(f"| {seq} | {row['frame']} | {'yes' if row['present'] else 'no'} | "+' | '.join(f'{x:.6f}' for x in vals)+' |')
lines+=['','Empty-frame Dice in the per-frame table is reported only as the mask agreement convention; use the false-alarm column above for empty-frame behavior. This is a local intervention, not an end-to-end memory-bank or video experiment. Ground-truth-prompt native and Lean results are not matched automatic-system baselines. The small feature-training set, previously inspected sequences, and unknown patient independence limit any generalization claim.','']
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'report_sha256':digest(out/'report.md'),'audit_sha256':{k:digest(a.root/f'{k}_audit_attempt01/results.json') for k in data},'runner_sha256':digest(Path(__file__))},indent=2)+'\n');print('\n'.join(lines))
