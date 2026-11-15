#!/usr/bin/env python3
"""Capture raw-window reads with verified source spans during a label-free gate."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_region_read_capture import capture_region_reads
from scripts.memory_token_io import digest
import scripts.run_lean_region_token_pool as runner


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--capture-targets',required=True);p.add_argument('--output',type=Path,required=True)
    a,rest=p.parse_known_args();targets={int(x) for x in a.capture_targets.split(',')}
    if not targets or min(targets)<2 or max(targets)>=15:raise ValueError('Seq3 gate target outside supported span')
    if '--strategy' not in rest or rest[rest.index('--strategy')+1]!='score_only' or '--mode' not in rest or rest[rest.index('--mode')+1]!='gate':
        raise ValueError('Capture requires score-only label-free gate')
    observations=[];original=runner.capture_region_scores
    @contextmanager
    def capture(model,scores):
        with original(model,scores):
            with capture_region_reads(model,targets,a.output,observations):yield
    runner.capture_region_scores=capture
    sys.argv=['run_lean_region_token_pool.py',*rest,'--output',str(a.output)]
    runner.main()
    if {e['frame'] for e in observations}!=targets:raise ValueError('Capture targets differ')
    entries=[]
    for event in observations:
        frame=event['frame'];p1=a.output/'snapshots'/event['snapshot_file'];p2=a.output/'source_snapshots'/f'{frame:06d}.pt'
        entries.append({'frame':frame,'read_sha256':digest(p1),'sources_sha256':digest(p2)})
    receipt={'passed':True,'targets':sorted(targets),'snapshots':entries,'records_sha256':digest(a.output/'records.json'),
             'pool_receipt_sha256':digest(a.output/'pool_observation.json'),'capture_sha256':digest(ROOT/'modeling/lean_region_read_capture.py'),
             'selector_sha256':digest(ROOT/'modeling/lean_region_read_intervention.py'),'runner_sha256':digest(Path(__file__))}
    (a.output/'region_capture_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
