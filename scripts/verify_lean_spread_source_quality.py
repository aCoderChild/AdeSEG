#!/usr/bin/env python3
"""Independently cross-check retained-frame and source-quality summary."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('analysis','audit','run','recent','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());d=read(a.analysis/'results.json');done=read(a.analysis/'completion.json');scores=read(a.audit/'results.json')
if not done['passed'] or done['results_sha256']!=digest(a.analysis/'results.json') or done['report_sha256']!=digest(a.analysis/'report.md') or d['audit_sha256']!=digest(a.audit/'results.json') or d['runner_sha256']!=digest(ROOT/'scripts/analyze_lean_spread_source_quality.py'):raise ValueError('Analysis source changed')
events={}
for arm,folder in (('recent',a.recent),('spread',a.run)):
 if d['event_sha256'][arm]!=digest(folder/'token_window_events.json'):raise ValueError('Event source changed')
 events[arm]={e['frame']:e['selected_frames'] for e in read(folder/'token_window_events.json') if e['stage']=='admission_plan'}
for target in d['rows']:
 i=target['target'];r=scores['frames'][i]
 if r['present']!=target['target_present']:raise ValueError('Target presence differs')
 for arm in ('recent','spread'):
  if target['target_scores'][arm]!=(r['scores'][arm]['dice'] if r['present'] else None):raise ValueError('Target score differs')
  entries=target['policies'][arm]
  if [e['frame'] for e in entries]!=events[arm][i-1]:raise ValueError('Retained frames differ')
  for e in entries:
   src=scores['frames'][e['frame']];s=src['scores'][arm]
   if e['age']!=i-e['frame'] or e['label_present']!=src['present'] or e['present_dice']!=(s['dice'] if src['present'] else None) or e['empty_false_alarm']!=s['false_alarm']:raise ValueError('Source score differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'analysis_sha256':digest(a.analysis/'results.json'),'verifier_sha256':digest(Path(__file__)),'targets':len(d['rows'])},indent=2)+'\n');print('All listed history indices, ages and saved-mask scores independently verified.')
