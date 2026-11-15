#!/usr/bin/env python3
"""Independently verify repeatable author-fusion fixed-read engineering gates."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.run_lean_offline_memory_ablation import equal
from modeling.lean_same_read_fusion import fused_skeleton


def read(p):return json.loads(p.read_text())
def check(ok,message):
    if not ok:raise ValueError(message)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    outcomes=[];snapshots=[]
    for i in (3,4):
        cap=a.root/f'capture_gate_attempt0{i}';run=a.root/f'replay_gate_attempt0{i}'
        c=read(cap/'fusion_capture_receipt.json');r=read(run/'completion.json')
        check(c['passed'] and c['entry']=='token' and c['mode']=='gate' and c['targets']==[8,12] and r['passed'],'Invalid capture or replay')
        check(c['capture_sha256']==digest(ROOT/'modeling/lean_same_read_fusion.py') and c['runner_sha256']==digest(ROOT/'scripts/capture_lean_same_read_fusion.py') and c['protocol_sha256']==digest(ROOT/'protocols/lean_same_read_fusion_v1.json'),'Capture code changed')
        check(r['runner_sha256']==digest(ROOT/'scripts/replay_lean_same_read_fusion.py') and r['model_capture_sha256']==c['capture_sha256'] and r['capture_sha256']==digest(cap/'fusion_capture_receipt.json'),'Replay source changed')
        for folder in (cap,run):
            integrity=read(folder/'parameter_integrity_after.json');check(integrity['unchanged'] and not integrity['changed_names'] and integrity['before']==integrity['after'],'Model parameters changed')
        this=[]
        for item,event in zip(c['snapshots'],r['events']):
            f=item['frame'];s=cap/'fusion_sources'/f'{f:06d}.pt';v=cap/'snapshots'/f'{f:06d}.pt'
            check(event['frame']==f and digest(s)==item['source_sha256'] and digest(v)==item['read_sha256'],'Snapshot identity mismatch')
            src=torch.load(s,map_location='cpu',weights_only=True);data=torch.load(v,map_location='cpu',weights_only=True)
            check(torch.equal(src['fused'],fused_skeleton(src['raw'],src['history'],src['alpha'])) and src['frame']==data['frame']==f,'Fusion mismatch')
            check(event['spatial_keys']+event['pointer_keys']==data['attention_inputs']['memory'].shape[0] and event['skeleton_tokens']==len(src['raw']) and event['raw_fused_mean_abs']>0,'No valid fixed-key fusion intervention')
            check(event['max_read_difference']['noop']==0 and event['max_read_difference']['fused_reconstructed']==0 and event['max_read_difference']['raw_skeleton']>0,'Control or replacement invalid')
            for method,h in event['masks'].items():check(digest(run/f'{f:06d}_{method}.png')==h,'Saved prediction changed')
            check(event['masks']['noop']==event['masks']['fused_reconstructed']==digest(cap/f'{f:06d}.png'),'Fused mask differs')
            this.append((src,data))
        snapshots.append(this);outcomes.append(r['events'])
    check(outcomes[0]==outcomes[1] and equal(snapshots[0],snapshots[1]),'Repeated gates differ')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'frames':[8,12],'capture_sha256':digest(ROOT/'modeling/lean_same_read_fusion.py'),
       'capture_runner_sha256':digest(ROOT/'scripts/capture_lean_same_read_fusion.py'),'replay_runner_sha256':digest(ROOT/'scripts/replay_lean_same_read_fusion.py'),
       'verifier_sha256':digest(Path(__file__)),'scope':'Label-free engineering only; no evidence of efficacy.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
