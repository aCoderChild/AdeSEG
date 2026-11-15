#!/usr/bin/env python3
"""Capture raw-two reads while reproducing its archived trajectory."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_capture import capture_memory_read
from scripts.memory_token_io import digest
import scripts.run_lean_raw_two_control as dispatcher
import scripts.run_lean_indexed_spatial_rope as indexed_runner
import scripts.run_seq23_viewport as crop_runner

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser(add_help=False)
for k in ('capture-targets','capture-reference','capture-gate-verification','output'):p.add_argument('--'+k,required=True)
a,rest=p.parse_known_args();reference=Path(a.capture_reference);out=Path(a.output);targets={int(x) for x in a.capture_targets.split(',')}
def val(flag):
 if flag not in rest:raise ValueError('Missing '+flag)
 return rest[rest.index(flag)+1]
entry=val('--entry');mode=val('--mode')
if entry=='token' and mode=='gate' and val('--sequence')=='seq3':engine=indexed_runner;limit=15;expected={8,12}
elif entry=='crop' and mode=='candidate':engine=crop_runner;limit=56;expected={8,20,35,48,51}
else:raise ValueError('Unsupported raw-two capture')
if targets!=expected:raise ValueError('Predeclared target frames changed')
prior=read(reference/'raw_two_control_receipt.json')
if not prior['passed'] or prior['entry']!=entry or prior['dispatcher_sha256']!=digest(ROOT/'scripts/run_lean_raw_two_control.py') or prior['protocol_sha256']!=digest(ROOT/'protocols/lean_raw_two_window_control_v1.json') or prior['delegate_receipt_sha256']!=digest(reference/'token_window_observation.json'):raise ValueError('Raw-two reference changed')
if entry=='crop':
 gate=read(Path(a.capture_gate_verification)/'verification.json')
 if not gate['passed'] or gate['runner_sha256']!=digest(Path(__file__)) or gate['capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or gate['protocol_sha256']!=digest(ROOT/'protocols/lean_seq23_cross_trajectory_memory_v1.json') or gate['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_raw_two_capture_gates.py'):raise ValueError('Capture gate changed')
old=engine.indexed_spatial_rope;events=[]
@contextmanager
def capture(model,positions):
 with old(model,positions):
  with capture_memory_read(model,targets,out,events):yield
engine.indexed_spatial_rope=capture
try:
 sys.argv=['run_lean_raw_two_control.py',*rest,'--output',str(out)];dispatcher.main()
finally:engine.indexed_spatial_rope=old
if {x['frame'] for x in events}!=targets:raise ValueError('Snapshot coverage differs')
files=('records.json','positions.json','selections.json','memory_events.json','token_window_events.json') if entry=='token' else ('prompt.json','prompt.png','frames.json','positions.json','selections.json','events.json','token_window_events.json')
if any(digest(reference/name)!=digest(out/name) for name in files):raise ValueError('Observer changed reference trajectory')
if entry=='crop' and any(digest(reference/f'{i:06d}.png')!=digest(out/f'{i:06d}.png') for i in range(limit)):raise ValueError('Observer changed saved mask')
snapshots=[{'frame':e['frame'],'read_sha256':digest(out/'snapshots'/e['snapshot_file'])} for e in events]
result={'passed':True,'entry':entry,'mode':mode,'targets':sorted(targets),'snapshots':snapshots,'viewport':entry=='crop','reference_receipt_sha256':digest(reference/'raw_two_control_receipt.json'),'output_receipt_sha256':digest(out/'raw_two_control_receipt.json'),'capture_sha256':digest(ROOT/'modeling/lean_memory_capture.py'),'protocol_sha256':digest(ROOT/'protocols/lean_seq23_cross_trajectory_memory_v1.json'),'runner_sha256':digest(Path(__file__))}
(out/'raw_two_capture_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
