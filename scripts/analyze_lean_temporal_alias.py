#!/usr/bin/env python3
"""Quantify ages that share the oldest learned temporal slot in verified runs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from modeling.lean_token_window_storage import temporal_slot
p=argparse.ArgumentParser();p.add_argument('--sequence',action='append',choices=('seq8','seq11','seq20','seq23'),required=True)
for k in ('spread-root','recent-root','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());results={}
if any(temporal_slot(100,100-age,7)!=(7-min(age,6)) for age in range(1,100)):raise ValueError('Unexpected temporal slot implementation')
for seq in a.sequence:
 verification=read(a.spread_root/f'{seq}_verification_attempt01'/'verification.json')
 if not verification['passed'] or verification['sequence']!=seq or verification['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_temporal_spread_run.py'):raise ValueError('Run verification differs')
 summary={}
 for arm,root in (('recent',a.recent_root),('spread',a.spread_root)):
  folder=root/f'{seq}_attempt01';ob=read(folder/'token_window_observation.json');path=folder/'token_window_events.json'
  if not ob['passed'] or ob['events_sha256']!=digest(path):raise ValueError('Event trace differs')
  plans=[e for e in read(path) if e['stage']=='admission_plan'];older_entries=entries=older_tokens=tokens=any_reads=0
  for plan in plans[:-1]:
   next_frame=plan['frame']+1;selected=plan['selected_frames'];any_reads+=any(next_frame-i>6 for i in selected);entries+=len(selected)
   if arm=='spread':sizes={x['frame']:x['tokens'] for x in plan['candidate_entries']}
   else:
    # The inherited runtime uses a 4096-token newly written grid and a
    # verified 1024-token compressed grid for each older retained entry.
    sizes={i:(plan['new_tokens'] if i==plan['frame'] else 1024) for i in selected}
   for i in selected:
    age=next_frame-i
    if age<1:raise ValueError('Non-past entry')
    n=sizes[i];tokens+=n
    if age>6:older_entries+=1;older_tokens+=n
  summary[arm]={'reads_with_age_over_6':any_reads,'reads_with_prior_admission':len(plans)-1,'older_entries':older_entries,'entries':entries,'older_tokens':older_tokens,'tokens':tokens,'event_sha256':digest(path)}
 results[seq]=summary
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
x={'scope':'Entries older than six frames all use the oldest learned nonconditioning temporal embedding. Counts exclude condition and insurance tokens; they measure temporal-slot aliasing, not utility. Recent-window counts assume its audited 1024-token compressed entries and full 4096-token newest grid.','sequences':results,'temporal_slot_sha256':digest(ROOT/'modeling/lean_token_window_storage.py'),'lean_assembly_sha256':digest(ROOT/'external/Lean-SAM2-token-window/sam2/modeling/sam2_base.py'),'runner_sha256':digest(Path(__file__))}
(out/'results.json').write_text(json.dumps(x,indent=2)+'\n');lines=['# Retained history beyond Lean’s learned temporal range','',x['scope'],'','| Sequence | Policy | Reads with age > 6 | Entries age > 6 | Represented tokens age > 6 |','| --- | --- | ---: | ---: | ---: |']
for seq,s in results.items():
 for arm,z in s.items():lines.append(f"| {seq} | {arm} | {z['reads_with_age_over_6']} / {z['reads_with_prior_admission']} | {z['older_entries']} / {z['entries']} | {z['older_tokens']} / {z['tokens']} |")
lines+=['','The positions of individual tokens and their feature contents can still differ. The shared temporal slot alone does not establish why segmentation changed. These are selected videos and temporal ages are measured at the next read.','']
(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
