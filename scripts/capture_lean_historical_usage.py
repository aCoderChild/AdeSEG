#!/usr/bin/env python3
"""Nest read-only Lean usage capture in the previously verified trajectory."""
import argparse,json,runpy,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import modeling.lean_memory_capture as memory_capture
from modeling.lean_historical_usage_capture import capture_historical_usage
from scripts.memory_token_io import digest
p=argparse.ArgumentParser(add_help=False)
for n in ('usage-output','usage-reference','usage-verification'):p.add_argument('--'+n,type=Path,required=True)
a,rest=p.parse_known_args();read=lambda f:json.loads(f.read_text())
def val(flag):
 if flag not in rest:raise ValueError('Missing '+flag)
 return rest[rest.index(flag)+1]
mode=val('--mode');seq=val('--sequence')
if mode=='gate' and seq!='seq3':raise ValueError('Gate must use seq3')
if mode=='candidate':
 v=read(a.usage_verification/'verification.json')
 if not v['passed'] or v['observer_sha256']!=digest(ROOT/'modeling/lean_historical_usage_capture.py') or v['runner_sha256']!=digest(Path(__file__)) or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_historical_usage.py') or v['protocol_sha256']!=digest(ROOT/'protocols/lean_historical_usage_capture_v1.json'):raise ValueError('Historical usage gate required')
elif mode!='gate':raise ValueError('Invalid mode')
prior=read(a.usage_reference/'utility_capture_receipt.json')
if not prior['passed'] or prior['sequence']!=seq:raise ValueError('Reference differs')
original=memory_capture.capture_memory_read;events=[]
@contextmanager
def nested(model,targets,output,observations):
 with original(model,targets,output,observations),capture_historical_usage(model,output,events):yield
memory_capture.capture_memory_read=nested;old=sys.argv
try:
 sys.argv=[str(ROOT/'scripts/capture_lean_utility_training_states.py'),*rest,'--output',str(a.usage_output)];runpy.run_path(sys.argv[0],run_name='__main__')
finally:memory_capture.capture_memory_read=original;sys.argv=old
own=read(a.usage_output/'utility_capture_receipt.json')
if own!=prior:raise ValueError('Usage observer changed frozen trajectory or reads')
for e in events:e['sha256']=digest(a.usage_output/'historical_usage'/e['file'])
receipt={'passed':True,'sequence':seq,'mode':mode,'events':events,'reference_sha256':digest(a.usage_reference/'utility_capture_receipt.json'),'nested_receipt_sha256':digest(a.usage_output/'utility_capture_receipt.json'),'observer_sha256':digest(ROOT/'modeling/lean_historical_usage_capture.py'),'runner_sha256':digest(Path(__file__)),'protocol_sha256':digest(ROOT/'protocols/lean_historical_usage_capture_v1.json')}
(a.usage_output/'historical_usage_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'passed':True,'sequence':seq,'usage_records':len(events)}))
