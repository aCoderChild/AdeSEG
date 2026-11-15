#!/usr/bin/env python3
"""Capture source encoder and following memory read in an observational replay."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_capture import capture_memory_read
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
for k in ('policy','entry','capture-reference','gate-verification','output'):p.add_argument('--'+k,required=True)
a,rest=p.parse_known_args();ref=Path(a.capture_reference);out=Path(a.output)
def val(flag):
 if flag not in rest:raise ValueError('Missing '+flag)
 return rest[rest.index(flag)+1]
mode=val('--mode')
if a.policy not in ('recent','spread'):raise ValueError('Unknown policy')
if a.entry=='gate' and mode=='gate' and val('--sequence')=='seq3':
 engine=indexed_runner;delegate=recent_token if a.policy=='recent' else spread_token;source={7,11};targets={8,12};limit=15;files=('records.json','positions.json','selections.json','memory_events.json','token_window_events.json')
elif a.entry=='crop' and mode=='candidate':
 engine=crop_runner;delegate=recent_crop if a.policy=='recent' else spread_crop;source={9,17,33,37,41};targets={10,18,34,38,42};limit=56;files=('prompt.json','prompt.png','frames.json','positions.json','selections.json','events.json','token_window_events.json')
 v=read(Path(a.gate_verification)/'verification.json')
 if not v['passed'] or v['runner_sha256']!=digest(Path(__file__)) or v['encoder_capture_sha256']!=digest(ROOT/'modeling/lean_memory_encoder_capture.py') or v['read_capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or v['protocol_sha256']!=digest(ROOT/'protocols/lean_encoder_empty_controls_v1.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_combined_capture_gates.py'):raise ValueError('Combined capture gates changed')
else:raise ValueError('Unsupported capture scope')
prior=read(ref/'token_window_observation.json')
if not prior['passed'] or prior['mode']!=mode or prior['sequence']!=('seq3' if a.entry=='gate' else 'seq23') or digest(ref/'token_window_events.json')!=prior['events_sha256']:raise ValueError('Frozen reference changed')
old=engine.indexed_spatial_rope;enc=[];reads=[]
@contextmanager
def capture(model,positions):
 with old(model,positions):
  with capture_memory_read(model,targets,out,reads),capture_memory_encoder(model,source,out,enc):yield
engine.indexed_spatial_rope=capture
try:
 sys.argv=[str(delegate.__file__),*rest,'--output',str(out)];delegate.main()
finally:engine.indexed_spatial_rope=old
if {x['frame'] for x in enc}!=source or {x['frame'] for x in reads}!=targets:raise ValueError('Snapshot coverage differs')
if any(digest(ref/name)!=digest(out/name) for name in files) or any(digest(ref/f'{i:06d}.png')!=digest(out/f'{i:06d}.png') for i in range(limit)):raise ValueError('Combined observers altered frozen trajectory')
encshots=[{'frame':e['frame'],'read_sha256':digest(out/'encoder_snapshots'/e['snapshot_file'])} for e in enc];readshots=[{'frame':e['frame'],'read_sha256':digest(out/'snapshots'/e['snapshot_file'])} for e in reads]
common={'passed':True,'policy':a.policy,'entry':a.entry,'mode':mode,'reference_observation_sha256':digest(ref/'token_window_observation.json'),'output_observation_sha256':digest(out/'token_window_observation.json'),'protocol_sha256':digest(ROOT/'protocols/lean_encoder_empty_controls_v1.json'),'runner_sha256':digest(Path(__file__))}
enc_receipt={**common,'targets':sorted(source),'snapshots':encshots,'module_sha256':digest(ROOT/'modeling/lean_memory_encoder_capture.py')}
read_receipt={**common,'targets':sorted(targets),'snapshots':readshots,'capture_sha256':digest(ROOT/'modeling/lean_memory_capture.py')}
(out/'encoder_capture_receipt.json').write_text(json.dumps(enc_receipt,indent=2)+'\n');name='recent_capture_receipt.json' if a.policy=='recent' else 'spread_capture_receipt.json';(out/name).write_text(json.dumps(read_receipt,indent=2)+'\n');print(json.dumps({'passed':True,'policy':a.policy,'sources':sorted(source),'targets':sorted(targets)}))
