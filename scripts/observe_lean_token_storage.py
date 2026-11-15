#!/usr/bin/env python3
"""Replay indexed Lean while observing its actual stored memory tensors."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_token_storage_observer import install_storage_observer
from scripts.memory_token_io import digest
import scripts.run_lean_indexed_spatial_rope as runner
PROTOCOL=ROOT/'protocols/lean_token_storage_observation_v1.json'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint','merged','reference','indexed-reference','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--sequence',required=True);p.add_argument('--label-free-gate',action='store_true');a=p.parse_args()
 if (a.label_free_gate and a.sequence!='seq3') or (not a.label_free_gate and a.sequence not in read(PROTOCOL)['cohort']):raise ValueError('Out-of-scope sequence')
 ref=read(a.indexed_reference/'records.json');done=read(a.indexed_reference/'completion.json')
 if not done['passed'] or digest(a.indexed_reference/'records.json')!=done['records_sha256']:raise ValueError('Indexed reference invalid')
 events=[];build=runner.build_author_predictor
 def observed_build(args,repo):
  model=build(args,repo);install_storage_observer(model,events);return model
 runner.build_author_predictor=observed_build
 args=['run_lean_indexed_spatial_rope.py','--mode','gate' if a.label_free_gate else 'candidate','--sequence',a.sequence]
 for key in ('checkpoint','merged','reference','cpu_verification','observed_reference','output'):args.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
 args.extend(['--gate-a',str(a.indexed_gate_a),'--gate-b',str(a.indexed_gate_b)]);sys.argv=args;runner.main()
 record=read(a.output/'records.json');keys=('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256')
 if not a.label_free_gate:keys+=('frames_sha256',)
 for key in keys:
  if record[key]!=ref[key]:raise ValueError(f'Observer changed reference: {key}')
 if not events:raise ValueError('No memory reads observed')
 (a.output/'storage_events.json').write_text(json.dumps(events,indent=2)+'\n')
 result={'passed':True,'sequence':a.sequence,'label_free_gate':a.label_free_gate,'events':len(events),
         'reference_records_sha256':done['records_sha256'],'replay_records_sha256':digest(a.output/'records.json'),
         'events_sha256':digest(a.output/'storage_events.json'),'protocol_sha256':digest(PROTOCOL),'source_sha256':digest(Path(__file__)),
         'observer_sha256':digest(ROOT/'modeling/lean_token_storage_observer.py')}
 (a.output/'storage_observation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
