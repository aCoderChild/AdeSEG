#!/usr/bin/env python3
"""Capture frozen author reads while exactly reproducing a spread trajectory."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_capture import capture_memory_read
from scripts.memory_token_io import digest
import scripts.run_lean_temporal_spread_window as token_runner
import scripts.run_seq23_temporal_spread_window as crop_dispatch
import scripts.run_lean_indexed_spatial_rope as indexed_runner
import scripts.run_seq23_viewport as crop_runner

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser(add_help=False)
for k in ('entry','capture-reference','capture-verification','output'):p.add_argument('--'+k,required=True)
a,rest=p.parse_known_args();reference=Path(a.capture_reference);out=Path(a.output)
def val(flag):
 if flag not in rest:raise ValueError('Missing '+flag)
 return rest[rest.index(flag)+1]
mode=val('--mode')
if a.entry=='gate' and mode=='gate' and val('--sequence')=='seq3':engine=indexed_runner;targets={8,12};delegate=token_runner;files=('records.json','positions.json','selections.json','memory_events.json','token_window_events.json');limit=15
elif a.entry=='crop' and mode=='candidate':
 engine=crop_runner;targets={8,20,35,48,51};delegate=crop_dispatch;files=('prompt.json','prompt.png','frames.json','positions.json','selections.json','events.json','token_window_events.json');limit=56
 v=read(Path(a.capture_verification)/'verification.json')
 if not v['passed'] or v['runner_sha256']!=digest(Path(__file__)) or v['capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or v['protocol_sha256']!=digest(ROOT/'protocols/lean_temporal_slot_rank_diagnostic_v1.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_temporal_spread_capture_gates.py'):raise ValueError('Capture engineering gates changed')
else:raise ValueError('Unsupported capture scope')
prior=read(reference/'token_window_observation.json')
if not prior['passed'] or prior['mode']!=mode or prior['sequence']!=('seq3' if a.entry=='gate' else 'seq23') or digest(reference/'token_window_events.json')!=prior['events_sha256']:raise ValueError('Frozen spread reference changed')
old=engine.indexed_spatial_rope;events=[]
@contextmanager
def capture(model,positions):
 with old(model,positions):
  with capture_memory_read(model,targets,out,events):yield
engine.indexed_spatial_rope=capture
try:
 sys.argv=[str(delegate.__file__),*rest,'--output',str(out)];delegate.main()
finally:engine.indexed_spatial_rope=old
if {x['frame'] for x in events}!=targets:raise ValueError('Snapshot coverage differs')
if any(digest(reference/name)!=digest(out/name) for name in files) or any(digest(reference/f'{i:06d}.png')!=digest(out/f'{i:06d}.png') for i in range(limit)):raise ValueError('Observer changed frozen trajectory')
snapshots=[{'frame':e['frame'],'read_sha256':digest(out/'snapshots'/e['snapshot_file'])} for e in events]
result={'passed':True,'entry':a.entry,'mode':mode,'targets':sorted(targets),'snapshots':snapshots,'reference_observation_sha256':digest(reference/'token_window_observation.json'),'output_observation_sha256':digest(out/'token_window_observation.json'),'capture_sha256':digest(ROOT/'modeling/lean_memory_capture.py'),'protocol_sha256':digest(ROOT/'protocols/lean_temporal_slot_rank_diagnostic_v1.json'),'runner_sha256':digest(Path(__file__))}
(out/'spread_capture_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'entry':a.entry,'snapshots':len(snapshots)}))
