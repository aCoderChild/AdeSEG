#!/usr/bin/env python3
"""Audit repeated, label-free fixed-read predictions and their scored outputs."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def check(ok,message):
    if not ok:raise ValueError(message)
def main():
    p=argparse.ArgumentParser()
    for key in ('capture','first','second','audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();cap=read(a.capture/'priority_capture_receipt.json');one=read(a.first/'completion.json');two=read(a.second/'completion.json')
    check(cap['passed'] and one['passed'] and one==two,'Replays differ or lack receipts')
    check(one['capture_sha256']==digest(a.capture/'priority_capture_receipt.json') and one['runner_sha256']==digest(ROOT/'scripts/replay_lean_region_priority_intervention.py') and one['rgb_module_sha256']==digest(ROOT/'modeling/lean_rgb_contrast_control.py'),'Implementation or capture changed')
    check(cap['priority_runner_sha256']==digest(ROOT/'scripts/capture_lean_region_priority_reads.py') and cap['capture_sha256']==digest(ROOT/'modeling/lean_region_read_capture.py') and cap['selector_sha256']==digest(ROOT/'modeling/lean_region_read_intervention.py'),'Capture code changed')
    check(digest(a.capture/'token_window_observation.json')==cap['observation_sha256'] and digest(a.capture/'region_source_scores.json')==cap['source_scores_sha256'],'Capture state changed')
    audit=read(a.audit/'completion.json');result=read(a.audit/'results.json')
    check(audit['passed'] and audit['results_sha256']==digest(a.audit/'results.json') and result['replay_completion_sha256']==digest(a.first/'completion.json') and result['source_sha256']==digest(ROOT/'scripts/audit_lean_region_priority_intervention.py'),'Score audit changed')
    methods={'usage','contrast','distributed','three_region','foreground','shuffled','rgb_contrast'}
    snapshots={item['frame']:item for item in cap['snapshots']};check(set(snapshots)==set(cap['targets']),'Target mismatch')
    for event in one['events']:
        f=event['frame'];item=snapshots[f];raw=a.capture/'snapshots'/f'{f:06d}.pt';src=a.capture/'source_snapshots'/f'{f:06d}.pt'
        check(digest(raw)==item['read_sha256'] and digest(src)==item['sources_sha256'],'Snapshot changed')
        s=torch.load(src,map_location='cpu',weights_only=True);eligible=set()
        for source in s['sources']:
            check(source['frame']<f-1,'Newest source included');eligible.update(range(source['start'],source['start']+len(source['token_ids'])))
        check(set(event['deleted'])==methods,'Missing method')
        for selection in event['deleted'].values():check(len(selection)==len(set(selection))==1024 and set(selection)<=eligible,'Deletion count or scope differs')
        for method,sha in event['masks'].items():
            check(digest(a.first/f'{f:06d}_{method}.png')==sha and digest(a.second/f'{f:06d}_{method}.png')==sha,'Prediction changed')
        check(event['masks']['noop']==digest(a.capture/f'{f:06d}.png'),'No-op mask differs')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'sequence':cap['sequence'],'frames':cap['targets'],'first_completion_sha256':digest(a.first/'completion.json'),
      'second_completion_sha256':digest(a.second/'completion.json'),'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__)),
      'scope':'Repeated, retrospective same-state deletions; not propagated video performance.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
