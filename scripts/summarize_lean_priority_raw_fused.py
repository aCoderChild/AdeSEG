#!/usr/bin/env python3
"""Summarize four frozen priority sequence audits without rerunning inference."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows={};sources={}
for seq in ('seq8','seq11','seq20','seq23'):
 folder=a.root/f'{seq}_audit_attempt01';done=json.loads((folder/'completion.json').read_text());result=json.loads((folder/'results.json').read_text())
 if not done['passed'] or done['results_sha256']!=digest(folder/'results.json') or done['source_sha256']!=digest(ROOT/('scripts/audit_lean_priority_raw_fused.py' if seq in ('seq8','seq20') else f'scripts/audit_{seq}_raw_two_control.py')):raise ValueError('Audit source differs')
 if seq in ('seq8','seq20'):
  check=json.loads((a.root/f'{seq}_audit_verification_attempt01'/'verification.json').read_text())
  if not check['passed'] or check['audit_sha256']!=done['results_sha256'] or check['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_priority_raw_fused.py'):raise ValueError('Independent verification differs')
 if result['matched_spatial_key_counts']!=result['read_count'] or result['matched_pointer_key_counts']!=result['read_count']:raise ValueError('Raw/fused read counts do not match')
 rows[seq]={k:{j:v[j] for j in ('matched_present_dice','matched_present_count','false_alarms','empty_count','mean_spatial_keys')} for k,v in result['conditions'].items()};rows[seq]['reads']=result['read_count'];sources[seq]=done['results_sha256']
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
data={'sequences':rows,'source_result_sha256':sources,'source_sha256':digest(Path(__file__)),'scope':'Selected automatic initialized sequences, previously inspected; present Dice uses historical target frames, false alarms cover all labeled-empty frames. Seq23 excludes the inset. GT-prompt comparisons have different prompts.'}
(out/'results.json').write_text(json.dumps(data,indent=2)+'\n')
lines=['# Four difficult sequences: matched-size raw versus fused memory','',data['scope'],'','| Sequence | Automatic indexed baseline | Raw token window | Fused window | Raw-two | Empty FAs: baseline / raw window / fused / raw-two | Matched raw-two / fused reads |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for seq,row in rows.items():
 arms=[row[k] for k in ('baseline','raw_window','fused_window','raw_two')];scores=' | '.join(f"{x['matched_present_dice']:.6f}" for x in arms);fa=' / '.join(str(x['false_alarms']) for x in arms);n=row['reads'];lines.append(f"| {seq} | {scores} | {fa} ({arms[0]['empty_count']}) | {n}/{n} spatial, {n}/{n} pointer |")
lines+=['','Raw-two and fused have equal read sizes on each listed read, but their memories, positional encodings and trajectories can differ. The full raw window usually reads more keys. These selected videos are exploratory and not independent evidence of generalization. Their initial prompts were automatic; sequence 20 frame 13 and sequence 23 frame 0 were labeled empty at initialization. Empty labels may be incomplete. The existing initialization schedule was reused, and no subsequent redetection or new confidence threshold was added.','', 'The represented spatial budget is not a total memory or latency measurement; no new runtime benchmark was performed. The frame IDs admitted by the existing token window matched six-frame recency on all four videos.','']
(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
