#!/usr/bin/env python3
"""Independently audit repeated fixed-read, label-free intervention gates."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.run_lean_offline_memory_ablation import equal


def read(p):return json.loads(p.read_text())
def check(ok,message):
    if not ok:raise ValueError(message)
def main():
    p=argparse.ArgumentParser();p.add_argument('--pool-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    gate=read(a.pool_root/'gate_verification_attempt01/verification.json');check(gate['passed'],'pool gates not verified')
    all_events=[];snapshots=[]
    for capnum,runnum in ((1,3),(2,4)):
        cap=a.pool_root/f'capture_gate_attempt0{capnum}';run=a.pool_root/f'offline_gate_attempt0{runnum}'
        receipt=read(cap/'region_capture_receipt.json');completion=read(run/'completion.json')
        check(receipt['passed'] and completion['passed'] and receipt['targets']==[8,12],'invalid capture or replay')
        check(receipt['runner_sha256']==digest(ROOT/'scripts/capture_lean_region_reads.py') and receipt['capture_sha256']==digest(ROOT/'modeling/lean_region_read_capture.py') and receipt['selector_sha256']==digest(ROOT/'modeling/lean_region_read_intervention.py'),'capture code changed')
        check(completion['runner_sha256']==digest(ROOT/'scripts/replay_lean_region_read_intervention.py') and completion['selector_sha256']==receipt['selector_sha256'] and completion['capture_sha256']==digest(cap/'region_capture_receipt.json'),'replay code or input changed')
        check(digest(cap/'records.json')==receipt['records_sha256'] and digest(cap/'pool_observation.json')==receipt['pool_receipt_sha256'],'capture files changed')
        integrity=read(run/'parameter_integrity_after.json');check(integrity['unchanged'] and integrity['before']==integrity['after'] and not integrity['changed_names'],'parameters changed')
        this=[]
        for event,item in zip(completion['events'],receipt['snapshots']):
            f=item['frame'];check(event['frame']==f,'frame mismatch')
            snapshot=cap/'snapshots'/f'{f:06d}.pt';source=cap/'source_snapshots'/f'{f:06d}.pt'
            check(digest(snapshot)==item['read_sha256'] and digest(source)==item['sources_sha256'],'snapshot changed')
            data=torch.load(snapshot,map_location='cpu',weights_only=True);sp=torch.load(source,map_location='cpu',weights_only=True)
            check(data['frame']==sp['frame']==f and sp['pointer_count']==data['attention_inputs']['num_obj_ptr_tokens'],'snapshot identity mismatch')
            eligible=set()
            for s in sp['sources']:
                check(s['frame']<f-1,'newest memory included')
                eligible.update(range(s['start'],s['start']+len(s['token_ids'])))
            for method,indices in event['deleted'].items():
                check(len(indices)==len(set(indices))==1024 and set(indices)<=eligible,'mismatched or unprotected intervention')
            check(set(event['deleted'])=={'usage','contrast','distributed','three_region','foreground','shuffled'},'missing control')
            for method,sha in event['masks'].items():check(digest(run/f'{f:06d}_{method}.png')==sha,'mask changed')
            check(event['masks']['noop']==digest(cap/f'{f:06d}.png'),'no-op mask changed')
            this.append((data,sp))
        all_events.append(completion['events']);snapshots.append(this)
    check(all_events[0]==all_events[1] and equal(snapshots[0],snapshots[1]),'independent captures or replays differ')
    a.output.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'targets':[8,12],'events_sha256':digest(a.pool_root/'offline_gate_attempt03/completion.json'),
      'capture_code_sha256':digest(ROOT/'modeling/lean_region_read_capture.py'),'replay_code_sha256':digest(ROOT/'scripts/replay_lean_region_read_intervention.py'),
      'verifier_sha256':digest(Path(__file__)),'interpretation':'Engineering on seq3 only; no utility or PolypGen efficacy assessed.'}
    (a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
