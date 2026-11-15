#!/usr/bin/env python3
"""Independently verify isolated Lean no-op and token-window gates."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.run_lean_temporal_spread_window import FORK,PROTOCOL,PATCHED
def read(p):return json.loads(p.read_text())
def inspect(path,variant,ref):
 o=read(path/'token_window_observation.json');r=read(path/'records.json');events=read(path/'token_window_events.json')
 if not o['passed'] or o['variant']!=variant or o['mode']!='gate' or o['sequence']!='seq3' or digest(path/'records.json')!=o['records_sha256'] or digest(path/'token_window_events.json')!=o['events_sha256'] or not read(path/'parameter_integrity_after.json')['unchanged']:
  raise ValueError('Gate receipt invalid')
 identity=o['identity']
 if identity['runner_sha256']!=digest(ROOT/'scripts/run_lean_temporal_spread_window.py') or identity['protocol_sha256']!=digest(PROTOCOL) or identity['runtime_sha256']!=digest(ROOT/'modeling/lean_temporal_spread_window.py') or identity['policy_sha256']!=digest(ROOT/'modeling/lean_temporal_spread_window.py') or identity['base_runtime_sha256']!=digest(ROOT/'modeling/lean_token_window_storage.py') or identity['base_policy_sha256']!=digest(ROOT/'modeling/token_memory_window.py') or identity['observer_sha256']!=digest(ROOT/'modeling/lean_token_storage_observer.py') or any(identity['patch_sha256'][k]!=digest(FORK/k) for k in PATCHED):raise ValueError('Gate source changed')
 reference=read(ref/'records.json');done=read(ref/'completion.json')
 if not done['passed'] or digest(ref/'records.json')!=done['records_sha256'] or o['reference_records_sha256']!=done['records_sha256']:raise ValueError('Reference changed')
 for i,h in enumerate(r['mask_sha256']):
  if digest(path/f'{i:06d}.png')!=h or digest(ref/f'{i:06d}.png')!=reference['mask_sha256'][i]:raise ValueError('Saved mask changed')
 if variant=='noop':
  if r!=reference or events:raise ValueError('No-op does not exactly reproduce reference')
  return o,r,events,{}
 if r['mask_sha256'][:3]!=reference['mask_sha256'][:3]:raise ValueError('Causal prefix differs')
 candidate_positions=read(path/'positions.json');reference_positions=read(ref/'positions.json')
 if len(candidate_positions)!=len(reference_positions) or [x['pointer_keys'] for x in candidate_positions]!=[x['pointer_keys'] for x in reference_positions]:raise ValueError('Pointer count trace changed')
 grouped={}
 for e in events:grouped.setdefault(e['frame'],[]).append(e)
 if sorted(grouped)!=list(range(1,len(r['mask_sha256']))):raise ValueError('Frame event coverage differs')
 evicted=[];maximum=0;previous_selected=[]
 expected=['after_read','before_admission','after_compression','admission_plan','after_write','after_bank_write']
 for frame,items in grouped.items():
  if [e['stage'] for e in items]!=expected:raise ValueError('Write/read event order changed')
  plan=items[3];write=items[4];bank=items[5]
  if plan['condition_tokens']!=4096 or plan['new_tokens']!=4096 or plan['budget']!=16384 or plan['insurance_reserve']!=3072:
   raise ValueError('Protected conditioning, new entry, or budget changed')
  if plan['planned_reserved_tokens']!=write['spatial_tokens']+3072 or plan['planned_reserved_tokens']>16384 or not plan['selected_frames'] or plan['selected_frames'][-1]!=frame:
   raise ValueError('Admission calculation does not match stored state')
  if plan['selected_frames']!=sorted(set(plan['selected_frames'])) or set(plan['evict_frames'])&set(plan['selected_frames']):raise ValueError('Invalid selected/evicted frame order')
  for e in (items[0],items[1],items[2],write,bank):
   if e['total_tokens']!=e['spatial_tokens']+e['insurance_tokens'] or e['total_tokens']>16384 or e['insurance_tokens']>3072 or e['spatial_tokens']+3072>16384:
    raise ValueError('Stored token cap or reserve exceeded')
   maximum=max(maximum,e['total_tokens'])
  remaining=[(x['frame'],x['tokens']) for x in plan['candidate_entries']]
  if not remaining or [x[0] for x in remaining[:-1]]!=previous_selected or remaining[-1][0]!=frame or len({x[0] for x in remaining})!=len(remaining):raise ValueError('Candidate sequence differs from previous resident set')
  cost=plan['condition_tokens']+plan['insurance_reserve']+sum(x[1] for x in remaining);removed=[]
  while cost>plan['budget']:
   options=[]
   for j,(idx,n) in enumerate(remaining[:-1]):
    prev=plan['condition_frame'] if j==0 else remaining[j-1][0]
    options.append((remaining[j+1][0]-prev,-idx,j))
   _,_,j=min(options);idx,n=remaining.pop(j);removed.append(idx);cost-=n
  if removed!=plan['evict_frames'] or [i for i,n in remaining]!=plan['selected_frames'] or cost!=plan['planned_reserved_tokens']:raise ValueError('Spread reconstruction differs')
  previous_selected=plan['selected_frames']
  evicted.extend(plan['evict_frames'])
 if not evicted or not any(e['stage']=='admission_plan' and min(e['selected_frames'])<e['frame']-5 for e in events):raise ValueError('Noncontiguous spread and eviction path not exercised')
 return o,r,events,{'max_tokens':maximum,'evictions':len(evicted),'changed_masks':sum(x!=y for x,y in zip(r['mask_sha256'],reference['mask_sha256'])),'pointer_trace':[x['pointer_keys'] for x in candidate_positions]}
def main():
 p=argparse.ArgumentParser()
 for k in ('noop-a','noop-b','window-a','window-b','reference','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();na,nr,ne,_=inspect(a.noop_a,'noop',a.reference);nb,nr2,ne2,_=inspect(a.noop_b,'noop',a.reference)
 wa,wr,we,summary=inspect(a.window_a,'window',a.reference);wb,wr2,we2,summary2=inspect(a.window_b,'window',a.reference)
 if na!=nb or nr!=nr2 or ne!=ne2 or wa!=wb or wr!=wr2 or we!=we2 or summary!=summary2 or na['identity']!=wa['identity']:raise ValueError('Independent gates differ')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'passed':True,'identity':wa['identity'],'noop_records_sha256':na['records_sha256'],'window_records_sha256':wa['records_sha256'],
         'window_events_sha256':wa['events_sha256'],'summary':summary,'verifier_sha256':digest(Path(__file__))}
 (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
