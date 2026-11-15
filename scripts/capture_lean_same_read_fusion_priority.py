#!/usr/bin/env python3
"""Capture author's raw and fused skeletons during an unchanged fused run."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_same_read_fusion import capture_fused_read
from scripts.memory_token_io import digest
import scripts.run_lean_fused_window_control as dispatcher
import scripts.run_lean_indexed_spatial_rope as indexed_runner
import scripts.run_seq23_viewport as crop_runner


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--capture-targets',required=True);p.add_argument('--capture-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a,rest=p.parse_known_args();targets={int(x) for x in a.capture_targets.split(',')}
    def val(flag):
        if flag not in rest:raise ValueError('Missing '+flag)
        return rest[rest.index(flag)+1]
    entry=val('--entry');mode=val('--mode')
    if entry=='token' and mode=='gate' and val('--sequence')=='seq3':engine=indexed_runner;limit=15
    elif entry=='token' and mode=='candidate' and val('--sequence') in ('seq8','seq11','seq20'):
        engine=indexed_runner;limit=int(read(a.capture_reference/'records.json')['frames'])
    elif entry=='crop' and mode=='candidate':engine=crop_runner;limit=56
    else:raise ValueError('Unsupported fusion capture')
    if not targets or min(targets)<3 or max(targets)>=limit:raise ValueError('Capture target out of range')
    prior=read(a.capture_reference/'fused_control_receipt.json')
    if not prior['passed'] or prior['entry']!=entry or prior['dispatcher_sha256']!=digest(ROOT/'scripts/run_lean_fused_window_control.py') or prior['delegate_receipt_sha256']!=digest(a.capture_reference/'token_window_observation.json'):
        raise ValueError('Reference fusion receipt invalid')
    if mode=='candidate':
        gate=Path(val('--same-read-gate-verification'));rest.remove('--same-read-gate-verification');rest.remove(str(gate))
        verification=read(gate/'verification.json')
        if not verification['passed'] or verification['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_same_read_fusion_gates.py') or verification['capture_sha256']!=digest(ROOT/'modeling/lean_same_read_fusion.py'):
            raise ValueError('Same-read engineering gates invalid')
    old=engine.indexed_spatial_rope;events=[]
    @contextmanager
    def capture(model,positions):
        with old(model,positions):
            with capture_fused_read(model,targets,a.output,events):yield
    engine.indexed_spatial_rope=capture
    try:
        sys.argv=['run_lean_fused_window_control.py',*rest,'--output',str(a.output)];dispatcher.main()
    finally:engine.indexed_spatial_rope=old
    if {x['frame'] for x in events}!=targets:raise ValueError('Missing captured reads')
    files=('records.json','positions.json','selections.json','memory_events.json','token_window_events.json') if entry=='token' else ('prompt.json','prompt.png','frames.json','positions.json','selections.json','events.json','token_window_events.json')
    if any(digest(a.capture_reference/name)!=digest(a.output/name) for name in files):raise ValueError('Observer changed baseline trace')
    if mode=='candidate' and any(digest(a.capture_reference/f'{i:06d}.png')!=digest(a.output/f'{i:06d}.png') for i in range(limit)):raise ValueError('Observer changed mask')
    snapshots=[]
    for e in events:
        f=e['frame'];snapshots.append({'frame':f,'read_sha256':digest(a.output/'snapshots'/e['snapshot_file']),'source_sha256':digest(a.output/'fusion_sources'/f'{f:06d}.pt')})
    result={'passed':True,'sequence':val('--sequence') if entry=='token' else 'seq23','entry':entry,'mode':mode,'targets':sorted(targets),'snapshots':snapshots,'viewport':entry=='crop',
      'reference_receipt_sha256':digest(a.capture_reference/'fused_control_receipt.json'),'output_receipt_sha256':digest(a.output/'fused_control_receipt.json'),
      'capture_sha256':digest(ROOT/'modeling/lean_same_read_fusion.py'),'protocol_sha256':digest(ROOT/'protocols/lean_same_read_fusion_priority_v1.json'),'runner_sha256':digest(Path(__file__))}
    (a.output/'fusion_capture_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
