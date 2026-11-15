#!/usr/bin/env python3
"""Verify a complete temporal-spread run against frozen gates and recent window."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.audit_lean_token_window import verify_events
from scripts.run_lean_temporal_spread_window import FORK,PATCHED,PROTOCOL

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--sequence',choices=('seq8','seq11','seq20','seq23'),required=True)
for k in ('run','recent','baseline','verification','crop-verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();o=read(a.run/'token_window_observation.json');done=read(a.run/'completion.json');v=read(a.verification/'verification.json');ident=v['identity'];r=read(a.recent/'token_window_observation.json');rd=read(a.recent/'completion.json')
if not all(x['passed'] for x in (o,done,v,r,rd)) or o['mode']!='candidate' or o['sequence']!=a.sequence or r['mode']!='candidate' or r['sequence']!=a.sequence or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_temporal_spread_gates.py') or ident['runner_sha256']!=digest(ROOT/'scripts/run_lean_temporal_spread_window.py') or ident['protocol_sha256']!=digest(PROTOCOL) or ident['runtime_sha256']!=digest(ROOT/'modeling/lean_temporal_spread_window.py') or ident['base_runtime_sha256']!=digest(ROOT/'modeling/lean_token_window_storage.py') or any(ident['patch_sha256'][k]!=digest(FORK/k) for k in PATCHED):raise ValueError('Run or verified code changed')
if digest(a.run/'token_window_events.json')!=o['events_sha256'] or digest(a.recent/'token_window_events.json')!=r['events_sha256'] or not read(a.run/'parameter_integrity_after.json')['unchanged'] or not read(a.recent/'parameter_integrity_after.json')['unchanged']:raise ValueError('Event trace or model parameters changed')
frames=read(a.run/'frames.json');recent=read(a.recent/'frames.json');events=read(a.run/'token_window_events.json');base=read(a.baseline/'completion.json')
if len(frames)!=len(recent) or not base['passed']:raise ValueError('Frame coverage differs')
if a.sequence=='seq23':
 cv=read(a.crop_verification/'verification.json')
 if not cv['passed'] or cv['verifier_sha256']!=digest(ROOT/'scripts/verify_seq23_temporal_spread_gates.py') or o['source_sha256']!=digest(ROOT/'scripts/run_seq23_temporal_spread_window.py') or o['verification_sha256']!=digest(a.verification/'verification.json') or o['baseline_completion_sha256']!=digest(a.baseline/'completion.json') or digest(a.run/'frames.json')!=done['frames_sha256'] or digest(a.recent/'frames.json')!=rd['frames_sha256'] or digest(a.run/'prompt.json')!=base['prompt_sha256'] or digest(a.recent/'prompt.json')!=base['prompt_sha256'] or digest(a.run/'prompt.png')!=digest(a.baseline/'prompt.png') or digest(a.recent/'prompt.png')!=digest(a.baseline/'prompt.png'):raise ValueError('Cropped prompt provenance differs')
 first=read(a.run/'prompt.json')['first_detection_index']
 for folder,completion in ((a.run,done),(a.recent,rd)):
  for key in ('positions','selections','setup'):
   if digest(folder/f'{key}.json')!=completion[key+'_sha256']:raise ValueError('Cropped trace differs')
else:
 rr=read(a.run/'records.json');old=read(a.recent/'records.json');br=read(a.baseline/'records.json')
 if o['identity']!=ident or o['records_sha256']!=digest(a.run/'records.json') or done['records_sha256']!=o['records_sha256'] or digest(a.recent/'records.json')!=rd['records_sha256'] or o['reference_records_sha256']!=digest(a.baseline/'records.json') or r['reference_records_sha256']!=digest(a.baseline/'records.json') or digest(a.baseline/'records.json')!=base['records_sha256']:raise ValueError('Reference records differ')
 for folder,records in ((a.run,rr),(a.recent,old)):
  for key in ('frames','positions','selections','memory_events'):
   if digest(folder/f'{key}.json')!=records[key+'_sha256']:raise ValueError('Run trace changed')
  if [x['mask_sha256'] for x in read(folder/'frames.json')]!=records['mask_sha256']:raise ValueError('Mask record differs')
 first=next(i for i in range(len(events)) if events[i]['stage']=='admission_plan');first=events[first]['condition_frame']
for i,(f,g) in enumerate(zip(frames,recent,strict=True)):
 if digest(a.run/f'{i:06d}.png')!=f['mask_sha256'] or digest(a.recent/f'{i:06d}.png')!=g['mask_sha256']:raise ValueError('Mask changed')
 if i<=first+2 and f['mask_sha256']!=g['mask_sha256']:raise ValueError('Causal prompt prefix differs')
recent_positions=read(a.recent/'positions.json');position=read(a.run/'positions.json')
if len(position)!=len(recent_positions):raise ValueError('Read count differs')
telemetry=verify_events(a.run,first,len(frames),recent_positions);matches=sum(x['spatial_keys']==y['spatial_keys'] for x,y in zip(position,recent_positions));pointers=sum(x['pointer_keys']==y['pointer_keys'] for x,y in zip(position,recent_positions))
plans=[x for x in events if x['stage']=='admission_plan'];prev=[];nonrecent=0;max_age=0
for z in plans:
 c=[(x['frame'],x['tokens']) for x in z['candidate_entries']];frame=z['frame']
 if [x[0] for x in c[:-1]]!=prev or c[-1][0]!=frame or c!=sorted(c):raise ValueError('Resident history does not match previous admission')
 used=z['condition_tokens']+z['insurance_reserve']+sum(n for _,n in c);removed=[]
 while used>z['budget']:
  options=[(c[j+1][0]-(z['condition_frame'] if j==0 else c[j-1][0]),-idx,j) for j,(idx,_) in enumerate(c[:-1])]
  if not options:raise ValueError('Protected entry cannot fit')
  _,_,j=min(options);idx,n=c.pop(j);removed.append(idx);used-=n
 selected=[idx for idx,_ in c]
 if selected!=z['selected_frames'] or removed!=z['evict_frames'] or used!=z['planned_reserved_tokens']:raise ValueError('Policy reconstruction differs')
 prev=selected;nonrecent+=bool(min(selected)<frame-5);max_age=max(max_age,frame-min(selected))
if not nonrecent:raise ValueError('No nonrecent history was retained')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
result={'passed':True,'sequence':a.sequence,'run_observation_sha256':digest(a.run/'token_window_observation.json'),'recent_observation_sha256':digest(a.recent/'token_window_observation.json'),'run_frames_sha256':digest(a.run/'frames.json'),'recent_frames_sha256':digest(a.recent/'frames.json'),'telemetry':telemetry,'read_count':len(position),'same_spatial_key_count_reads':matches,'same_pointer_key_count_reads':pointers,'admissions':len(plans),'nonrecent_admissions':nonrecent,'max_retained_age':max_age,'first_prompt':first,'verifier_sha256':digest(Path(__file__))}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
