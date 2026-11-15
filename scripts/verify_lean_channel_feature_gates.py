#!/usr/bin/env python3
"""Verify feature-observer no-op gates and two independent feature captures."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_feature_capture import sha
from scripts.memory_token_io import digest,validate_output
from scripts.run_lean_offline_memory_ablation import equal


def read(p):return json.loads(p.read_text())
def check(ok,message):
    if not ok:raise ValueError(message)
def main():
    p=argparse.ArgumentParser()
    for key in ('first','second','reference','preflight','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();manifest=read(a.preflight/'manifest.json');done=read(a.preflight/'completion.json')
    check(done['passed'] and done['manifest_sha256']==digest(a.preflight/'manifest.json') and manifest['protocol_sha256']==digest(ROOT/'protocols/lean_channel_attention_training_pilot_v1.json'),'Preflight changed')
    prior=read(a.reference/'pool_observation.json');expected=manifest['sequences']['seq3']['targets'];outcomes=[];samples=[]
    for folder in (a.first,a.second):
        r=read(folder/'feature_capture_receipt.json');record=read(folder/'records.json');pool=read(folder/'pool_observation.json');integrity=read(folder/'parameter_integrity_after.json')
        check(r['passed'] and r['sequence']=='seq3' and r['targets']==expected and pool['passed'] and pool['strategy']=='score_only','Receipt mismatch')
        check(r['capture_sha256']==digest(ROOT/'modeling/lean_channel_feature_capture.py') and r['runner_sha256']==digest(ROOT/'scripts/capture_lean_channel_feature_gate.py') and r['preflight_sha256']==digest(a.preflight/'manifest.json') and r['reference_receipt_sha256']==digest(a.reference/'pool_observation.json') and r['output_receipt_sha256']==digest(folder/'pool_observation.json'),'Source changed')
        check(integrity['unchanged'] and integrity['before']==integrity['after'] and not integrity['changed_names'],'Model weights changed')
        for name in ('records.json','pool_events.json','source_scores.json','positions.json','selections.json','memory_events.json'):
            check(digest(folder/name)==digest(a.reference/name),'Baseline trajectory differs: '+name)
        check(pool['records_sha256']==digest(folder/'records.json') and prior['records_sha256']==digest(a.reference/'records.json'),'Records changed')
        check(record['mask_sha256']==[digest(folder/f'{i:06d}.png') for i in range(record['frames'])],'Mask changed')
        group=[]
        for item in r['features']:
            frame=item['frame'];path=folder/'features'/f'{frame:06d}.pt';check(digest(path)==item['file_sha256'],'Feature file changed')
            v=torch.load(path,map_location='cpu',weights_only=True)
            check(v['frame']==frame and v['feature'].shape==(32,256,256) and v['foreground'].shape==v['background'].shape==(32,),'Feature geometry differs')
            for key in ('feature','foreground','background'):check(sha(v[key])==item[key+'_sha256'] and torch.isfinite(v[key]).all(),'Feature hash differs')
            group.append(v)
        check({x['frame'] for x in r['features']}==set(expected),'Coverage differs')
        outcomes.append(r);samples.append(group)
    check(outcomes[0]==outcomes[1] and equal(samples[0],samples[1]),'Feature replays differ')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'sequence':'seq3','targets':expected,'preflight_sha256':digest(a.preflight/'manifest.json'),'capture_sha256':digest(ROOT/'modeling/lean_channel_feature_capture.py'),
      'gate_runner_sha256':digest(ROOT/'scripts/capture_lean_channel_feature_gate.py'),'verifier_sha256':digest(Path(__file__)),'scope':'Feature-capture no-op only; no trained model or label-based inference.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
