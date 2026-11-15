#!/usr/bin/env python3
"""Run exact indexed Lean replay with discarded local memory ablations."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.lean_memory_shadow import memory_shadow_observer
from scripts.memory_token_io import digest
import scripts.run_lean_indexed_spatial_rope as runner
PROTOCOL=ROOT/'protocols/lean_memory_shadow_ablation_v1.json'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint','merged','reference','indexed-reference','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--sequence',required=True);p.add_argument('--label-free-gate',action='store_true');p.add_argument('--empty-test',action='store_true');a=p.parse_args()
 if (a.label_free_gate and a.sequence!='seq3') or (not a.label_free_gate and a.sequence not in read(PROTOCOL)['cohort']):raise ValueError('Out-of-scope sequence')
 ref=read(a.indexed_reference/'records.json');done=read(a.indexed_reference/'completion.json')
 if not done['passed'] or digest(a.indexed_reference/'records.json')!=done['records_sha256']:raise ValueError('Indexed reference invalid')
 source=Path(read(a.merged/'locations.json')[a.sequence]);old=read(source/'records.json')[0];folder=source/a.sequence
 if digest(folder/'prompt.json')!=old['prompt_sha256']:raise ValueError('Original prompt changed')
 first=read(folder/'prompt.json')['first_detection_index'];frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}[a.sequence]
 target=sorted(set([first+1]+[first+round((len(frames)-1-first)*i/4) for i in range(1,5)]))
 if not target or target[0]<=first or target[-1]>=len(frames):raise ValueError('Invalid fixed frame schedule')
 if a.empty_test:
  if not a.label_free_gate:raise ValueError('Empty-target test is engineering-only')
  target=[]
 events=[];base_context=runner.indexed_spatial_rope
 @contextmanager
 def with_shadow(model,positions):
  with base_context(model,positions):
   with memory_shadow_observer(model,positions,set(target),a.output,events):yield
 runner.indexed_spatial_rope=with_shadow
 args=['run_lean_indexed_spatial_rope.py','--mode','gate' if a.label_free_gate else 'candidate','--sequence',a.sequence]
 for key in ('checkpoint','merged','reference','cpu_verification','observed_reference','output'):args.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
 args.extend(['--gate-a',str(a.indexed_gate_a),'--gate-b',str(a.indexed_gate_b)]);sys.argv=args
 runner.main();r=read(a.output/'records.json');keys=('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256')
 if not a.label_free_gate:keys+=('frames_sha256',)
 for key in keys:
  if r[key]!=ref[key]:raise ValueError(f'Shadow changed live reference: {key}')
 if [e['frame'] for e in events]!=target:raise ValueError('Shadow frame coverage differs')
 for event in events:
  frame=event['frame'];base=a.output/'shadow'/f'{frame:06d}_baseline.png';noop=a.output/'shadow'/f'{frame:06d}_noop.png'
  for kind,h in event['mask_sha256'].items():
   if digest(a.output/'shadow'/f'{frame:06d}_{kind}.png')!=h:raise ValueError('Shadow mask changed')
  if not np.array_equal(np.asarray(Image.open(base))>0,np.asarray(Image.open(a.indexed_reference/f'{frame:06d}.png'))>0) or digest(base)!=digest(noop):
   raise ValueError('Shadow baseline does not reproduce saved video mask')
 (a.output/'shadow_events.json').write_text(json.dumps(events,indent=2)+'\n')
 result={'passed':True,'sequence':a.sequence,'label_free_gate':a.label_free_gate,'first_detection_index':first,'target_frames':target,
  'reference_records_sha256':done['records_sha256'],'replay_records_sha256':digest(a.output/'records.json'), 'shadow_events_sha256':digest(a.output/'shadow_events.json'),
  'protocol_sha256':digest(PROTOCOL),'source_sha256':digest(Path(__file__)),'observer_sha256':digest(ROOT/'modeling/lean_memory_shadow.py')}
 (a.output/'shadow_observation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
