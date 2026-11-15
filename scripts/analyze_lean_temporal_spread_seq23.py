#!/usr/bin/env python3
"""Describe pre-existing late-recovery interval after the spread run is frozen."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('audit','audit-verification','run','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());d=read(a.audit/'results.json');done=read(a.audit/'completion.json');v=read(a.audit_verification/'verification.json')
if not done['passed'] or not v['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or v['audit_sha256']!=done['results_sha256'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_temporal_spread_audit.py') or d['sequence']!='seq23':raise ValueError('Source audit differs')
plans={x['frame']:x for x in read(a.run/'token_window_events.json') if x['stage']=='admission_plan'};ranges=[]
for lo,hi in ((0,46),(47,55)):
 rows=[x for x in d['frames'] if x['matched'] and lo<=x['frame']<=hi]
 if not rows:raise ValueError('No matched frames')
 r={'start':lo,'end':hi,'matched_present_count':len(rows)}
 for name in ('recent','spread'):r[name]=float(np.mean([x['scores'][name]['dice'] for x in rows]))
 ranges.append(r)
samples={str(i):plans[i]['selected_frames'] for i in (20,35,47,48,51,55)}
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);payload={'scope':'Post-hoc intervals previously identified from the same sequence; correlated frames and no independent validation. Selection IDs after each listed frame are not an isolated causal ablation.','ranges':ranges,'selected_frame_ids_after_write':samples,'audit_sha256':digest(a.audit/'results.json'),'runner_sha256':digest(Path(__file__))}
(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n');lines=['# Sequence 23: spread versus recent, descriptive intervals','',payload['scope'],'','| Frames | Matched-present frames | Recent Dice | Spread Dice |','| --- | ---: | ---: | ---: |']
for x in ranges:lines.append(f"| {x['start']}–{x['end']} | {x['matched_present_count']} | {x['recent']:.6f} | {x['spread']:.6f} |")
lines+=['','After-write retained history IDs (includes the newest frame):']
for k,ids in samples.items():lines.append(f"- frame {k}: {ids}")
lines+=['','These intervals and frames were chosen after earlier inspection. Memory contents, history ages and propagated predictions vary together; the results do not establish why the later recovery was lost.','']
(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
