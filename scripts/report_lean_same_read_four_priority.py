#!/usr/bin/env python3
"""Summarize audited same-read raw/fused substitutions on priority sequences."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('priority','seq23','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());rows=[];sources={}
for seq in ('seq8','seq11','seq20','seq23'):
 folder=a.seq23 if seq=='seq23' else a.priority/f'{seq}_audit_attempt01';done=read(folder/'completion.json');r=read(folder/'results.json')
 if not done['passed'] or done['results_sha256']!=digest(folder/'results.json') or done['report_sha256']!=digest(folder/'report.md') or r['sequence']!=seq:raise ValueError('Audit changed')
 if seq!='seq23':
  v=read(a.priority/f'{seq}_verification_attempt01/verification.json')
  if not v['passed'] or v['results_sha256']!=digest(folder/'results.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_same_read_fusion_priority_audit.py'):raise ValueError('Independent score verification missing')
 else:
  if r['audit_sha256']!=digest(ROOT/'scripts/audit_lean_same_read_fusion.py'):raise ValueError('Seq23 audit changed')
 sources[seq]=digest(folder/'results.json')
 for e in r['frames']:
  f=e['scores']['noop'];z=e['scores']['raw_skeleton'];rows.append({'sequence':seq,'frame':e['frame'],'present':e['present'],'fused_dice':f['dice'],'raw_dice':z['dice'],'dice_difference':z['dice']-f['dice'],'fused_false_alarm':f['false_alarm'],'raw_false_alarm':z['false_alarm'],'changed_pixels':e['raw_changed_pixels'],'spatial_keys':e['spatial_keys'],'pointer_keys':e['pointer_keys']})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'rows':rows,'source_sha256':sources,'runner_sha256':digest(Path(__file__))};(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Same-read raw versus fused: priority sequences','','Each row replaces the author-fused skeleton with its matching raw source at the same frozen read. Query, other memory, positions, pointer tokens and key count are unchanged. No result is fed into later frames. Seq23 excludes the inset.','','| Sequence | Frame | Label | Fused Dice | Raw Dice | Δ Dice | Changed pixels | Empty false alarms fused / raw |','|---|---:|---|---:|---:|---:|---:|---|']
for e in rows:
 alarm='—' if e['present'] else f"{e['fused_false_alarm']} / {e['raw_false_alarm']}";lines.append(f"| {e['sequence']} | {e['frame']} | {'present' if e['present'] else 'empty'} | {e['fused_dice']:.6f} | {e['raw_dice']:.6f} | {e['dice_difference']:+.6f} | {e['changed_pixels']} | {alarm} |")
lines+=['','These frames and videos were previously inspected, and empty labels may be incomplete. This isolates one content substitution conditional on the saved state; it does not measure autoregressive propagation or explain trajectory-level differences. Large changed-pixel counts with zero Dice can occur when neither mask overlaps the annotation. The author’s existing fusion reliability weighting is retained; no new gate, threshold or redetection was added.']
(out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
