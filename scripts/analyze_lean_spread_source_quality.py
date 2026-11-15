#!/usr/bin/env python3
"""Describe labeled quality of historical entries before selected seq23 reads."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('audit','audit-verification','run','recent','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());d=read(a.audit/'results.json');done=read(a.audit/'completion.json');v=read(a.audit_verification/'verification.json')
if not done['passed'] or not v['passed'] or d['sequence']!='seq23' or done['results_sha256']!=digest(a.audit/'results.json') or v['audit_sha256']!=done['results_sha256'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_temporal_spread_audit.py'):raise ValueError('Verified audit differs')
events={};receipts={}
for arm,folder in (('recent',a.recent),('spread',a.run)):
 receipt=read(folder/'token_window_observation.json')
 if not receipt['passed'] or digest(folder/'token_window_events.json')!=receipt['events_sha256']:raise ValueError('Memory event trace changed')
 events[arm]={e['frame']:e for e in read(folder/'token_window_events.json') if e['stage']=='admission_plan'};receipts[arm]=digest(folder/'token_window_events.json')
rows=d['frames'];outrows=[]
for target in (47,48,51,55):
 item={'target':target,'target_present':rows[target]['present'],'target_scores':{k:rows[target]['scores'][k]['dice'] if rows[target]['present'] else None for k in ('recent','spread')},'policies':{}}
 for arm in ('recent','spread'):
  # The previous write determines retained nonconditioning entries at the next read.
  selected=events[arm][target-1]['selected_frames'];sources=[]
  for i in selected:
   r=rows[i];score=r['scores'][arm]
   sources.append({'frame':i,'age':target-i,'label_present':r['present'],'present_dice':score['dice'] if r['present'] else None,'empty_false_alarm':score['false_alarm']})
  item['policies'][arm]=sources
 outrows.append(item)
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
payload={'scope':'Post-hoc description of nonconditioning spatial memory entries after the previous write. It excludes separate insurance-bank and conditioning entries; labels are used only in this analysis. A selected frame’s saved mask is not a direct measure of memory-token utility.','rows':outrows,'audit_sha256':digest(a.audit/'results.json'),'event_sha256':receipts,'runner_sha256':digest(Path(__file__))}
(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n');lines=['# Sequence 23: retained source-frame quality before selected reads','',payload['scope'],'']
for x in outrows:
 lines.append(f"## Target frame {x['target']}: recent / spread Dice {x['target_scores']['recent']:.4f} / {x['target_scores']['spread']:.4f}")
 for name,items in x['policies'].items():
  bits=[]
  for s in items:
   status=(f"present Dice {s['present_dice']:.3f}" if s['label_present'] else f"empty, false alarm {s['empty_false_alarm']}")
   bits.append(f"{s['frame']} (age {s['age']}; {status})")
  lines.append(f"- {name}: "+'; '.join(bits))
 lines.append('')
lines+=['These frames and policies were selected after observing the outcome. Similar or inaccurate source masks do not prove how the network uses individual tokens or why propagation changed. Empty annotations may be incomplete.','']
(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
