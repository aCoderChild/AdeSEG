#!/usr/bin/env python3
"""Validate and summarize exact-replay Lean stored-token observations."""
import argparse,collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from modeling.token_memory_window import SpatialMemory,select_token_window
PROTOCOL=ROOT/'protocols/lean_token_storage_observation_v1.json'
def read(p):return json.loads(p.read_text())
def check(path,ref):
 obs=read(path/'storage_observation.json');done=read(ref/'completion.json')
 if not obs['passed'] or not done['passed'] or digest(ref/'records.json')!=done['records_sha256'] or obs['reference_records_sha256']!=done['records_sha256']:
  raise ValueError('Reference invalid')
 if obs['protocol_sha256']!=digest(PROTOCOL) or obs['source_sha256']!=digest(ROOT/'scripts/observe_lean_token_storage.py') or obs['observer_sha256']!=digest(ROOT/'modeling/lean_token_storage_observer.py'):
  raise ValueError('Observer identity changed')
 if digest(path/'records.json')!=obs['replay_records_sha256'] or digest(path/'storage_events.json')!=obs['events_sha256'] or not read(path/'parameter_integrity_after.json')['unchanged']:
  raise ValueError('Replay or events changed')
 x,y=read(path/'records.json'),read(ref/'records.json')
 for key in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256'):
  if x[key]!=y[key]:raise ValueError('Replay differs')
 for index,checksum in enumerate(x['mask_sha256']):
  if digest(path/f'{index:06d}.png')!=checksum or digest(ref/f'{index:06d}.png')!=checksum:
   raise ValueError('Saved replay or reference mask changed')
 if not obs['label_free_gate'] and x['frames_sha256']!=y['frames_sha256']:raise ValueError('Frame data differs')
 events=read(path/'storage_events.json')
 if len(events)!=obs['events']:raise ValueError('Events incomplete')
 for event in events:
  for phase in ('before','after'):
   s=event[phase]
   if sum(x['feature_tokens'] for x in s['entries'])!=s['spatial_feature_tokens'] or sum(x['author_logical_tokens'] for x in s['entries'])!=s['logical_spatial_tokens'] or sum(s['insurance_entries'])!=s['insurance_feature_tokens']:
    raise ValueError('Stored count mismatch')
   if any(e['feature_tokens']!=e['position_tokens'] for e in s['entries']):raise ValueError('Alignment mismatch')
 return obs,events
def main():
 p=argparse.ArgumentParser()
 for key in ('root','indexed-root','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();sp=a.indexed_root
 ga,ea=check(a.root/'gate_attempt01',sp/'indexed_gate_attempt01');gb,eb=check(a.root/'gate_attempt02',sp/'indexed_gate_attempt01')
 if not ga['label_free_gate'] or not gb['label_free_gate'] or ea!=eb:raise ValueError('Independent gates differ')
 summaries={}
 for name in read(PROTOCOL)['cohort']:
  candidates=list(sp.glob(f'part_*_attempt01/{name}'))
  if len(candidates)!=1:raise ValueError('Missing reference sequence')
  obs,events=check(a.root/f'{name}_attempt01',candidates[0])
  if obs['sequence']!=name or obs['label_free_gate']:raise ValueError('Observation sequence differs')
  phases={}
  for phase in ('before','after'):
   snapshots=[e[phase] for e in events if not e['initial']]
   if not snapshots:raise ValueError('Missing non-initial reads')
   pairs=collections.Counter((x['spatial_feature_tokens'],x['logical_spatial_tokens'],x['insurance_feature_tokens']) for x in snapshots)
   phases[phase]={'reads':len(snapshots),'physical_range':[min(x['spatial_feature_tokens'] for x in snapshots),max(x['spatial_feature_tokens'] for x in snapshots)],
     'insurance_range':[min(x['insurance_feature_tokens'] for x in snapshots),max(x['insurance_feature_tokens'] for x in snapshots)],
     'pointer_elements_range':[min(x['object_pointer_elements'] for x in snapshots),max(x['object_pointer_elements'] for x in snapshots)],
     'output_entries_range':[min(len(x['entries']) for x in snapshots),max(len(x['entries']) for x in snapshots)],
     'max_represented_plus_insurance':max(x['spatial_feature_tokens']+x['insurance_feature_tokens'] for x in snapshots),
     'physical_exceeds_author_logical':sum(x['spatial_feature_tokens']>x['logical_spatial_tokens'] for x in snapshots),
     'common_counts':[{'spatial':k[0],'logical':k[1],'insurance':k[2],'reads':v} for k,v in pairs.most_common(5)]}
   if phase=='before':
    # Offline what-if on already resident entries only: this does not simulate
    # a changed history, predictor state, EMA, or future storage writes.
    hypothetical={}
    for budget in (12288,16384):
     checked=unchanged=impossible=0; dropped=0
     for s in snapshots:
      cond=sum(e['feature_tokens'] for e in s['entries'] if e['group']=='cond_frame_outputs')
      history=[SpatialMemory(e['frame'],e['feature_tokens']) for e in s['entries'] if e['group']=='non_cond_frame_outputs' and e['feature_tokens']]
      if not cond:continue
      checked+=1
      try:selected,_=select_token_window(cond,history,budget,reserved_tokens=s['insurance_feature_tokens'])
      except ValueError:impossible+=1;continue
      dropped_here=len(history)-len(selected);dropped+=dropped_here
      if not dropped_here:unchanged+=1
     hypothetical[str(budget)]={'reads_checked':checked,'all_resident_history_fits':unchanged,'protected_allocation_does_not_fit':impossible,'summed_resident_entries_excluded':dropped}
    phases[phase]['offline_resident_snapshot_what_if']=hypothetical
  summaries[name]={'events':len(events),'phases':phases,'observation_sha256':digest(a.root/f'{name}_attempt01/storage_observation.json')}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'passed':True,'scope':'Read-time representation counts, not unique bytes or a total storage guarantee.','sequences':summaries,'gate_events_sha256':ga['events_sha256'],'audit_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
