#!/usr/bin/env python3
"""Validate and summarize read-only Lean attention usage replays."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def describe(values):
 if not values:return None
 a=np.asarray(values,dtype=float);return {'mean':float(np.mean(a)),'median':float(np.median(a)),'minimum':float(np.min(a)),'maximum':float(np.max(a))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--part',type=Path,action='append',required=True)
 for name in ('merged','indexed-root','verification','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();v=read(a.verification);protocol=read(ROOT/'protocols/lean_attention_mass_diagnostic_v1.json')
 if (not v['passed'] or v['source_sha256']!=digest(ROOT/'scripts/verify_lean_attention_mass.py')
     or v['observer_sha256']!=digest(ROOT/'scripts/observe_lean_attention_mass.py') or v['protocol_sha256']!=digest(ROOT/'protocols/lean_attention_mass_diagnostic_v1.json')):raise ValueError('Engineering verification changed')
 merge=read(a.merged/'completion.json');loc=read(a.merged/'locations.json');baseline=read(a.indexed_root/'audited_cohort_attempt01/completion.json')
 if not merge['passed'] or digest(a.merged/'results.json')!=merge['results_sha256'] or not baseline['passed'] or digest(a.indexed_root/'audited_cohort_attempt01/results.json')!=baseline['results_sha256']:raise ValueError('Reference invalid')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};results={}
 for part in a.part:
  setup,done=read(part/'setup.json'),read(part/'completion.json')
  if not done['passed'] or done['sequences']!=setup['sequences'] or setup['verification_sha256']!=digest(a.verification) or setup['runner_sha256']!=v['observer_sha256']:raise ValueError('Partition invalid')
  for name in setup['sequences']:
   if name in results:raise ValueError('Overlapping partitions')
   run=part/name;o=read(run/'attention_observation.json');r=read(run/'records.json');rdone=read(run/'completion.json');events=read(run/'attention_events.json')
   if (not o['passed'] or o['sequence']!=name or o['label_free_gate'] or o['source_sha256']!=v['observer_sha256'] or o['protocol_sha256']!=v['protocol_sha256']
       or o['summary_source_sha256']!=digest(ROOT/'modeling/lean_attention_mass.py') or o['attention_events_sha256']!=digest(run/'attention_events.json')
       or len(events)!=o['events'] or not rdone['passed'] or digest(run/'records.json')!=rdone['records_sha256'] or o['replay_records_sha256']!=rdone['records_sha256']
       or not read(run/'parameter_integrity_after.json')['unchanged']):raise ValueError('Observation invalid')
   refs=[x/name for x in (a.indexed_root/'part_a_attempt01',a.indexed_root/'part_b_attempt01') if (x/name/'completion.json').exists()]
   if len(refs)!=1:raise ValueError('Missing reference')
   ref=refs[0];rr=read(ref/'records.json');refdone=read(ref/'completion.json')
   if not refdone['passed'] or digest(ref/'records.json')!=refdone['records_sha256'] or o['reference_records_sha256']!=refdone['records_sha256']:raise ValueError('Reference changed')
   for key in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256','frames_sha256'):
    if r[key]!=rr[key]:raise ValueError('Observer changed reference')
   for i,h in enumerate(r['mask_sha256']):
    if digest(run/f'{i:06d}.png')!=h or digest(ref/f'{i:06d}.png')!=h:raise ValueError('Video mask changed')
   positions=read(run/'positions.json');selections=read(run/'selections.json')
   if len(positions)!=len(events) or len(selections)!=o['selection_events']:raise ValueError('Trace coverage mismatch')
   selection_keys={(x['frame_index'],x['source_frame_index']) for x in selections}
   event_keys={(e['frame'],e['latest_full_memory']['source_frame']) for e in events if e['latest_full_memory'] is not None}
   if len(selection_keys)!=len(selections) or event_keys!=selection_keys:raise ValueError('Selection event alignment differs')
   source=Path(loc[name]);oldrec=read(source/'records.json')[0];old_done=read(source/'completion.json');oldfolder=source/name
   if not old_done['passed'] or digest(source/'records.json')!=old_done['records_sha256'] or digest(oldfolder/'frames.json')!=oldrec['frames_sha256'] or digest(oldfolder/'prompt.json')!=oldrec['prompt_sha256']:raise ValueError('Source changed')
   oldrows=read(oldfolder/'frames.json');frames=videos[name];first=read(oldfolder/'prompt.json')['first_detection_index']
   if [x['frame'] for x in events]!=list(range(first+1,len(frames))):raise ValueError('Frame coverage differs')
   labeled=[]
   for e,pos in zip(events,positions):
    fractions=[e['conditioning_mass'],e['other_spatial_mass'],e['pointer_mass']]
    if (e['conditioning_keys']!=4096 or e['keys']!=pos['spatial_keys']+pos['pointer_keys'] or e['pointer_keys']!=pos['pointer_keys']
        or any(not math.isfinite(x) or x<0 or x>1 for x in fractions) or abs(sum(fractions)-1)>1e-5):raise ValueError('Attention segment differs')
    i=e['frame'];frame=frames[i]
    if digest(frame.image_path)!=oldrows[i]['image_sha256'] or digest(frame.mask_path)!=oldrows[i]['label_sha256']:raise ValueError('Input changed')
    present=bool((load_label_mask(frame.mask_path,[1])==1).any())
    latest=e['latest_full_memory']
    if latest is not None and (latest['end']-latest['start']!=4096 or latest['start']<4096 or latest['end']!=e['keys']-e['pointer_keys'] or not 0<=latest['retained_mass_at_this_read']<=latest['mass_before_pruning']<=1):raise ValueError('Latest segment invalid')
    labeled.append({'frame':i,'present':present,**{k:e[k] for k in ('keys','conditioning_mass','other_spatial_mass','pointer_mass','latest_full_memory')}})
   def summarize(items):
    return {'frames':len(items),'conditioning_mass':describe([x['conditioning_mass'] for x in items]),'other_spatial_mass':describe([x['other_spatial_mass'] for x in items]),
     'pointer_mass':describe([x['pointer_mass'] for x in items]),'latest_before_pruning_mass':describe([x['latest_full_memory']['mass_before_pruning'] for x in items if x['latest_full_memory']]),
     'retained_mass_at_that_read':describe([x['latest_full_memory']['retained_mass_at_this_read'] for x in items if x['latest_full_memory']])}
   results[name]={'first_detection_index':first,'all':summarize(labeled),'labeled_present':summarize([x for x in labeled if x['present']]),'labeled_empty':summarize([x for x in labeled if not x['present']]),'frames':labeled}
 if set(results)!=set(protocol['cohort']):raise ValueError('Cohort incomplete')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);summary={'scope':'Exact reference replays, last-layer attention averaged over queries, heads and batch; condition memory verified by exact position vectors. Labels used only for retrospective present/empty grouping.','sequences':results}
 (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Lean attention usage: exact replay diagnostic','',summary['scope'],'','| Sequence | Observed frames | Conditioning mass mean / median | Other spatial mass mean | Pointer mass mean | Latest full-memory mass mean before pruning | Retained subset mass at that read |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for name,r in results.items():
  x=r['all'];fmt=lambda k:'—' if x[k] is None else f"{x[k]['mean']:.3f}"
  cm=x['conditioning_mass'];lines.append(f"| {name} | {x['frames']} | {cm['mean']:.3f} / {cm['median']:.3f} | {fmt('other_spatial_mass')} | {fmt('pointer_mass')} | {fmt('latest_before_pruning_mass')} | {fmt('retained_mass_at_that_read')} |")
 lines += ['', 'The last two columns refer to the full latest memory at the read where Lean decides what to retain; they do not measure that memory after pruning. Attention mass is not causal contribution and cannot establish that conditioning or other memory caused an error. This is a selected, previously inspected cohort; no model intervention or new threshold was used.','']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'source_sha256':digest(Path(__file__))},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
