#!/usr/bin/env python3
"""Report verified matched conditioning-region interventions."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('root','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();seqs=('seq8','seq11','seq20','seq23');methods=('noop','foreground','background','boundary','uniform','shuffled','attention_oracle');records={};events={}
for seq in seqs:
 audit=a.root/f'{seq}_audit_attempt01';v=read(a.root/f'{seq}_audit_verification_attempt01'/'verification.json');r=read(a.root/f'{seq}_replay_attempt01'/'completion.json');d=read(audit/'results.json');done=read(audit/'completion.json')
 if not all(x['passed'] for x in (v,r,done)) or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_condition_region_audit.py') or v['audit_sha256']!=digest(audit/'results.json') or done['results_sha256']!=v['audit_sha256'] or any(set(x['scores'])!=set(methods) for x in d['frames']):raise ValueError('Audit incomplete')
 records[seq]=d;events[seq]={x['frame']:x for x in r['events']}
lines=['# Same-read conditioning-region deletion','','On four previously inspected frozen reads per sequence, each intervention deletes 1,024 of the original 4,096 conditioning keys. Other memory, pointer tokens and indexed rotary positions remain unchanged. All deletion arms use equal total key counts; the no-op uses 1,024 additional keys and is a descriptive reference. The region scores come from the automatic initial mask, not labels: foreground and background refer to that predicted mask and may be incorrect. Uniform and shuffled controls use deterministic spatial ordering; the attention-oracle arm ranks using last-layer usage from the original full read and is not a single-pass deployable strategy.','','Two independent replays and independent selection, position, attention-mass and mask audits passed. Sequence 23 used the inset-excluded viewport. These are post-hoc local deletion effects, with changed attention normalization, not video results or evidence for pruning prompt memory.','','| Sequence | Present / empty | Deleted conditioning region | Present mean Dice | Empty false alarms | Removed pre-deletion attention mass | Mean automatic-mask occupancy |','| --- | ---: | --- | ---: | ---: | ---: | ---: |']
for seq in seqs:
 rows=records[seq]['frames'];pos=[x for x in rows if x['present']];neg=[x for x in rows if not x['present']];ev=list(events[seq].values())
 for method in methods:
  dice=f"{np.mean([x['scores'][method]['dice'] for x in pos]):.6f}" if pos else '—';fa=sum(bool(x['scores'][method]['false_alarm']) for x in neg)
  mass='—' if method=='noop' else f"{np.mean([x['removed_attention_mass'][method] for x in ev]):.2%}"
  occ='—' if method=='noop' else f"{np.mean([x['mean_prompt_occupancy'][method] for x in ev]):.3f}"
  lines.append(f'| {seq} | {len(pos)} / {len(neg)} | {method} | {dice} | {fa} / {len(neg)} | {mass} | {occ} |')
lines+=['','## Per-frame results','','| Sequence | Frame | Label | Full read | Foreground removed | Background removed | Uniform removed | Shuffled removed | Attention oracle removed |','| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for seq in seqs:
 for x in records[seq]['frames']:
  vals=[x['scores'][m]['dice'] for m in ('noop','foreground','background','uniform','shuffled','attention_oracle')]
  lines.append(f"| {seq} | {x['frame']} | {'present' if x['present'] else 'empty'} | "+' | '.join(f'{v:.6f}' for v in vals)+' |')
lines+=['','On labeled-empty frames, the table uses the conventional empty-mask Dice; false alarms are summarized separately above. Attention mass is not a measure of causal importance. Prior full-video conditioning-memory reduction substantially worsened sequence 8; this diagnostic does not supersede that result. Patient independence is unknown.','']
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'report_sha256':digest(out/'report.md'),'runner_sha256':digest(Path(__file__))},indent=2)+'\n');print('\n'.join(lines))
