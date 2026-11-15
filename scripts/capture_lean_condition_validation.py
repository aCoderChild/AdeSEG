#!/usr/bin/env python3
"""Capture non-training sequence reads while reproducing saved trajectories."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_capture import capture_memory_read
from scripts.memory_token_io import digest
from scripts.run_lean_offline_memory_ablation import equal
import scripts.capture_lean_channel_features as runner

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser(add_help=False)
for k in ('preflight','saved-feature-reference','snapshot-gate','capture-verification','output'):p.add_argument('--'+k,type=Path,required=True)
a,rest=p.parse_known_args()
if '--sequence' not in rest or '--mode' not in rest:raise ValueError('Missing sequence or mode')
seq=rest[rest.index('--sequence')+1];mode=rest[rest.index('--mode')+1];manifest=read(a.preflight/'manifest.json');pre=read(a.preflight/'completion.json')
if not pre['passed'] or pre['manifest_sha256']!=digest(a.preflight/'manifest.json') or manifest['source_sha256']!=digest(ROOT/'scripts/preflight_lean_channel_training.py'):raise ValueError('Preflight changed')
if (mode=='gate' and seq!='seq3') or (mode=='candidate' and (seq not in ('seq9','seq10') or seq not in manifest['groups']['validation'])):raise ValueError('Sequence outside predeclared scope')
if mode=='candidate':
 v=read(a.capture_verification/'verification.json')
 if not v['passed'] or v['capture_runner_sha256']!=digest(Path(__file__)) or v['memory_capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or v['preflight_sha256']!=digest(a.preflight/'manifest.json') or v['protocol_sha256']!=digest(ROOT/'protocols/lean_condition_region_validation_v1.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_condition_validation_capture_gate.py'):raise ValueError('Capture gate changed')
old_gate=read(a.snapshot_gate/'region_capture_receipt.json');saved=read(a.saved_feature_reference/'channel_feature_receipt.json')
if not old_gate['passed'] or old_gate['capture_sha256']!=digest(ROOT/'modeling/lean_region_read_capture.py') or old_gate['runner_sha256']!=digest(ROOT/'scripts/capture_lean_region_reads.py') or not saved['passed'] or saved['sequence']!=seq or saved['preflight_sha256']!=digest(a.preflight/'manifest.json') or saved['runner_sha256']!=digest(ROOT/'scripts/capture_lean_channel_features.py') or saved['records_sha256']!=digest(a.saved_feature_reference/'records.json'):raise ValueError('Verified source changed')
if mode=='gate':targets=set(old_gate['targets'])
else:
 m=manifest['sequences'][seq];targets={i for i in m['targets'] if i>=m['first_prompt']+7}
if not targets:raise ValueError('No preselected targets with adequate history')
observations=[];original=runner.capture_channel_features
@contextmanager
def capture(model,feature_targets,output,feature_observations):
 with original(model,feature_targets,output,feature_observations):
  with capture_memory_read(model,targets,output,observations):yield
runner.capture_channel_features=capture
try:
 sys.argv=[str(ROOT/'scripts/capture_lean_channel_features.py'),*rest,'--preflight',str(a.preflight),'--output',str(a.output)];runner.main()
finally:runner.capture_channel_features=original
new=read(a.output/'channel_feature_receipt.json');r0=read(a.saved_feature_reference/'records.json');r1=read(a.output/'records.json')
for key in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256','frames_sha256'):
 if r0[key]!=r1[key]:raise ValueError('Capture changed trajectory: '+key)
if digest(a.output/'token_window_events.json')!=digest(a.saved_feature_reference/'token_window_events.json') or new['pipeline']!=saved['pipeline'] or new['features']!=saved['features']:raise ValueError('Feature capture or token trace changed')
if {o['frame'] for o in observations}!=targets:raise ValueError('Read coverage changed')
snapshots=[]
for o in observations:
 i=o['frame'];path=a.output/'snapshots'/o['snapshot_file']
 if mode=='gate':
  old=next(x for x in old_gate['snapshots'] if x['frame']==i);old_path=a.snapshot_gate/'snapshots'/f'{i:06d}.pt'
  if digest(old_path)!=old['read_sha256'] or not equal(torch.load(path,map_location='cpu',weights_only=True),torch.load(old_path,map_location='cpu',weights_only=True)):raise ValueError('Engineering snapshot differs')
 snapshots.append({'frame':i,'read_sha256':digest(path)})
receipt={'passed':True,'mode':mode,'sequence':seq,'targets':sorted(targets),'snapshots':snapshots,'feature_reference_sha256':digest(a.saved_feature_reference/'channel_feature_receipt.json'),'records_sha256':digest(a.output/'records.json'),'preflight_sha256':digest(a.preflight/'manifest.json'),'memory_capture_sha256':digest(ROOT/'modeling/lean_memory_capture.py'),'runner_sha256':digest(Path(__file__)),'protocol_sha256':digest(ROOT/'protocols/lean_condition_region_validation_v1.json')}
(a.output/'condition_validation_capture.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'passed':True,'mode':mode,'sequence':seq,'targets':sorted(targets)}))
