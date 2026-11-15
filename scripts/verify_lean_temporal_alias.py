#!/usr/bin/env python3
"""Recompute reported old-history counts from recorded admissions."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('analysis','spread-root','recent-root','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());done=read(a.analysis/'completion.json');d=read(a.analysis/'results.json')
if not done['passed'] or done['results_sha256']!=digest(a.analysis/'results.json') or done['report_sha256']!=digest(a.analysis/'report.md') or d['runner_sha256']!=digest(ROOT/'scripts/analyze_lean_temporal_alias.py') or d['temporal_slot_sha256']!=digest(ROOT/'modeling/lean_token_window_storage.py') or d['lean_assembly_sha256']!=digest(ROOT/'external/Lean-SAM2-token-window/sam2/modeling/sam2_base.py'):raise ValueError('Analysis or source changed')
for seq,arms in d['sequences'].items():
 for name,root in (('recent',a.recent_root),('spread',a.spread_root)):
  path=root/f'{seq}_attempt01'/'token_window_events.json'
  if digest(path)!=arms[name]['event_sha256']:raise ValueError('Events changed')
  plans=[x for x in read(path) if x['stage']=='admission_plan'];entries=[];reads=0
  for plan in plans[:-1]:
   ids=plan['selected_frames'];next_frame=plan['frame']+1;reads+=any(next_frame-i>6 for i in ids)
   sizes={x['frame']:x['tokens'] for x in plan['candidate_entries']} if name=='spread' else {i:(plan['new_tokens'] if i==plan['frame'] else 1024) for i in ids}
   # Check that the assumed entry sizes reproduce the recorded protected budget.
   if plan['condition_tokens']+plan['insurance_reserve']+sum(sizes[i] for i in ids)!=plan['planned_reserved_tokens']:raise ValueError('Per-entry sizes do not reproduce budget')
   entries.extend((next_frame-i,sizes[i]) for i in ids)
  z=arms[name];expected={'reads_with_age_over_6':reads,'reads_with_prior_admission':len(plans)-1,'older_entries':sum(age>6 for age,n in entries),'entries':len(entries),'older_tokens':sum(n for age,n in entries if age>6),'tokens':sum(n for age,n in entries)}
  if any(z[k]!=value for k,value in expected.items()):raise ValueError('Temporal count differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'analysis_sha256':digest(a.analysis/'results.json'),'sequences':list(d['sequences']),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Temporal age and budget counts independently verified.')
