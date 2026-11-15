#!/usr/bin/env python3
"""Nest existing verified feature/read observers for sequence-separated utility data."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_capture import capture_memory_read
from scripts.memory_token_io import digest
import scripts.capture_lean_channel_features as base

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser(add_help=False)
for k in ('nested-reference','nested-verification','output'):p.add_argument('--'+k,type=Path,required=True)
a,rest=p.parse_known_args()
def val(flag):
 if flag not in rest:raise ValueError('Missing '+flag)
 return rest[rest.index(flag)+1]
mode=val('--mode');seq=val('--sequence');ref=a.nested_reference;prior=read(ref/'channel_feature_receipt.json')
if not prior['passed'] or prior['sequence']!=seq or prior['runner_sha256']!=digest(ROOT/'scripts/capture_lean_channel_features.py'):raise ValueError('Feature reference invalid')
if mode=='candidate':
 v=read(a.nested_verification/'verification.json')
 if not v['passed'] or v['runner_sha256']!=digest(Path(__file__)) or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_utility_training_gates.py') or v['protocol_sha256']!=digest(ROOT/'protocols/lean_utility_training_capture_v1.json'):raise ValueError('Nested engineering verification missing')
elif mode!='gate' or seq!='seq3':raise ValueError('Invalid engineering scope')
events=[];original=base.capture_channel_features
@contextmanager
def nested(model,targets,output,observations):
 with original(model,targets,output,observations),capture_memory_read(model,targets,output,events):yield
base.capture_channel_features=nested
try:
 sys.argv=[str(base.__file__),*rest,'--output',str(a.output)];base.main()
finally:base.capture_channel_features=original
own=read(a.output/'channel_feature_receipt.json')
if own['targets']!=prior['targets'] or own['records_sha256']!=prior['records_sha256'] or own['events_sha256']!=prior['events_sha256'] or own['feature_pipeline_sha256']!=prior['feature_pipeline_sha256']:raise ValueError('Nested observer altered trajectory')
for old,new in zip(prior['features'],own['features'],strict=True):
 if old!=new or digest(ref/'features'/f"{old['frame']:06d}.pt")!=digest(a.output/'features'/f"{old['frame']:06d}.pt"):raise ValueError('Nested observer altered features')
if {x['frame'] for x in events}!=set(own['targets']):raise ValueError('Read snapshot coverage differs')
shots=[{'frame':x['frame'],'sha256':digest(a.output/'snapshots'/x['snapshot_file'])} for x in events]
receipt={'passed':True,'mode':mode,'sequence':seq,'targets':own['targets'],'snapshots':shots,'feature_receipt_sha256':digest(a.output/'channel_feature_receipt.json'),'reference_receipt_sha256':digest(ref/'channel_feature_receipt.json'),'protocol_sha256':digest(ROOT/'protocols/lean_utility_training_capture_v1.json'),'runner_sha256':digest(Path(__file__)),'read_observer_sha256':digest(ROOT/'modeling/lean_memory_capture.py')}
(a.output/'utility_capture_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'passed':True,'sequence':seq,'snapshots':len(shots)}))
