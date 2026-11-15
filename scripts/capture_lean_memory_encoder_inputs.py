#!/usr/bin/env python3
"""Capture encoder arguments while exactly reproducing frozen token-window videos."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_encoder_capture import capture_memory_encoder
from scripts.memory_token_io import digest
import scripts.run_lean_token_window as recent_token
import scripts.run_lean_temporal_spread_window as spread_token
import scripts.run_seq23_token_window as recent_crop
import scripts.run_seq23_temporal_spread_window as spread_crop
import scripts.run_lean_indexed_spatial_rope as indexed_runner
import scripts.run_seq23_viewport as crop_runner

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser(add_help=False)
for k in ('policy','entry','capture-reference','capture-verification','output'):p.add_argument('--'+k,required=True)
a,rest=p.parse_known_args();ref=Path(a.capture_reference);out=Path(a.output)
def val(flag):
 if flag not in rest:raise ValueError('Missing '+flag)
 return rest[rest.index(flag)+1]
mode=val('--mode')
if a.policy not in ('recent','spread'):raise ValueError('Unknown policy')
if a.entry=='gate' and mode=='gate' and val('--sequence')=='seq3':
 engine=indexed_runner;delegate=recent_token if a.policy=='recent' else spread_token;targets={7,11};limit=15;files=('records.json','positions.json','selections.json','memory_events.json','token_window_events.json')
elif a.entry=='crop' and mode=='candidate':
 engine=crop_runner;delegate=recent_crop if a.policy=='recent' else spread_crop;targets={47,50};limit=56;files=('prompt.json','prompt.png','frames.json','positions.json','selections.json','events.json','token_window_events.json')
 v=read(Path(a.capture_verification)/'verification.json')
 if not v['passed'] or v['runner_sha256']!=digest(Path(__file__)) or v['module_sha256']!=digest(ROOT/'modeling/lean_memory_encoder_capture.py') or v['protocol_sha256']!=digest(ROOT/'protocols/lean_newest_memory_component_v1.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_memory_encoder_capture_gates.py'):raise ValueError('Encoder capture gates changed')
else:raise ValueError('Unsupported capture scope')
prior=read(ref/'token_window_observation.json')
if not prior['passed'] or prior['mode']!=mode or prior['sequence']!=('seq3' if a.entry=='gate' else 'seq23') or digest(ref/'token_window_events.json')!=prior['events_sha256']:raise ValueError('Frozen reference changed')
old=engine.indexed_spatial_rope;events=[]
@contextmanager
def capture(model,positions):
 with old(model,positions):
  with capture_memory_encoder(model,targets,out,events):yield
engine.indexed_spatial_rope=capture
try:
 sys.argv=[str(delegate.__file__),*rest,'--output',str(out)];delegate.main()
finally:engine.indexed_spatial_rope=old
if {x['frame'] for x in events}!=targets:raise ValueError('Snapshot coverage differs')
if any(digest(ref/name)!=digest(out/name) for name in files) or any(digest(ref/f'{i:06d}.png')!=digest(out/f'{i:06d}.png') for i in range(limit)):raise ValueError('Observer altered frozen trajectory')
snapshots=[{'frame':e['frame'],'read_sha256':digest(out/'encoder_snapshots'/e['snapshot_file'])} for e in events]
result={'passed':True,'policy':a.policy,'entry':a.entry,'mode':mode,'targets':sorted(targets),'snapshots':snapshots,'reference_observation_sha256':digest(ref/'token_window_observation.json'),'output_observation_sha256':digest(out/'token_window_observation.json'),'module_sha256':digest(ROOT/'modeling/lean_memory_encoder_capture.py'),'protocol_sha256':digest(ROOT/'protocols/lean_newest_memory_component_v1.json'),'runner_sha256':digest(Path(__file__))}
(out/'encoder_capture_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'policy':a.policy,'entry':a.entry,'targets':sorted(targets)}))
