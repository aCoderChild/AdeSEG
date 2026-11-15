#!/usr/bin/env python3
"""Reproduce a priority-sequence raw window while capturing fixed reads."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_region_read_capture import capture_region_reads
from modeling.lean_region_token_pool import capture_region_scores
from scripts.memory_token_io import digest
import scripts.run_lean_token_window as raw_runner
import scripts.run_seq23_token_window as crop_window
import scripts.run_lean_indexed_spatial_rope as indexed_runner
import scripts.run_seq23_viewport as crop_runner


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--capture-sequence',choices=('seq11','seq23'),required=True)
    p.add_argument('--capture-targets',required=True);p.add_argument('--raw-reference',type=Path,required=True)
    p.add_argument('--offline-gate-verification',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a,rest=p.parse_known_args();targets={int(x) for x in a.capture_targets.split(',')}
    limits={'seq11':228,'seq23':56}
    if not targets or min(targets)<3 or max(targets)>=limits[a.capture_sequence]:raise ValueError('Capture target outside sequence')
    verification=read(a.offline_gate_verification/'verification.json')
    if not verification['passed'] or verification['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_region_read_gates.py') or verification['capture_code_sha256']!=digest(ROOT/'modeling/lean_region_read_capture.py') or verification['replay_code_sha256']!=digest(ROOT/'scripts/replay_lean_region_read_intervention.py'):
        raise ValueError('Offline engineering verification changed')
    if '--mode' not in rest or rest[rest.index('--mode')+1]!='candidate':raise ValueError('Priority capture requires candidate mode')
    if a.capture_sequence=='seq11':
        if '--sequence' not in rest or rest[rest.index('--sequence')+1]!='seq11' or '--variant' not in rest or rest[rest.index('--variant')+1]!='window':raise ValueError('Expected raw sequence 11 window')
        driver=raw_runner;engine=indexed_runner
    else:driver=crop_window;engine=crop_runner
    old=engine.indexed_spatial_rope;observations=[];scores=[]
    @contextmanager
    def capture(model,positions):
        with old(model,positions):
            with capture_region_scores(model,scores):
                with capture_region_reads(model,targets,a.output,observations):yield
    engine.indexed_spatial_rope=capture
    sys.argv=[driver.__file__,*rest,'--output',str(a.output)]
    driver.main()
    if {e['frame'] for e in observations}!=targets:raise ValueError('Capture coverage differs')
    reference=a.raw_reference
    if a.capture_sequence=='seq11':
        orig=read(reference/'records.json');new=read(a.output/'records.json')
        for key in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256','frames_sha256'):
            if orig[key]!=new[key]:raise ValueError('Capture changed reference: '+key)
    else:
        for name in ('prompt.json','prompt.png','frames.json','positions.json','selections.json','events.json','token_window_events.json'):
            if digest(reference/name)!=digest(a.output/name):raise ValueError('Cropped capture changed reference: '+name)
        for i in range(limits[a.capture_sequence]):
            if digest(reference/f'{i:06d}.png')!=digest(a.output/f'{i:06d}.png'):raise ValueError('Cropped capture mask changed')
    snapshots=[]
    for event in observations:
        f=event['frame'];snapshots.append({'frame':f,'read_sha256':digest(a.output/'snapshots'/event['snapshot_file']),
          'sources_sha256':digest(a.output/'source_snapshots'/f'{f:06d}.pt')})
    (a.output/'region_source_scores.json').write_text(json.dumps(scores,indent=2)+'\n')
    result={'passed':True,'sequence':a.capture_sequence,'viewport':a.capture_sequence=='seq23','targets':sorted(targets),'snapshots':snapshots,
      'reference':str(reference),'reference_observation_sha256':digest(reference/'token_window_observation.json'),
      'observation_sha256':digest(a.output/'token_window_observation.json'),'source_scores_sha256':digest(a.output/'region_source_scores.json'),
      'capture_sha256':digest(ROOT/'modeling/lean_region_read_capture.py'),'selector_sha256':digest(ROOT/'modeling/lean_region_read_intervention.py'),
      'priority_runner_sha256':digest(Path(__file__)),'offline_verification_sha256':digest(a.offline_gate_verification/'verification.json')}
    (a.output/'priority_capture_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
